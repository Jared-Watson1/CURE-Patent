import os
import json
from pathlib import Path
import time
import requests
from lxml import etree
import csv
import pandas as pd
import pinecone
from dotenv import load_dotenv
import openai
from embedding import createEmbeddings, process_chunk
from vector_storage import chunks
import concurrent.futures
from datetime import timedelta, date, datetime
import xml.etree.ElementTree as ET

load_dotenv()
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
csvFile = "data/pm_eutils.csv"
input_datapath = csvFile
output_embeddings_path = "data/pm_embeddings_with_id.csv"
embedding_encoding = "utf-8"
embedding_model = "text-embedding-ada-002"
max_tokens = 4096
top_n = 1000


# Function to send a request to the NCBI API
def send_request(query, retstart, retmax, mindate=None, maxdate=None):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retstart": retstart,
        "usehistory": "y",
        "api_key": NCBI_API_KEY,
    }

    if mindate and maxdate:
        params["mindate"] = mindate
        params["maxdate"] = maxdate
        params["datetype"] = "pdat"

    response = requests.get(BASE_URL + "esearch.fcgi", params=params)
    return response


# Function to generate date ranges
def daterange(start_date, end_date, delta_days=30):
    for n in range(0, int((end_date - start_date).days), delta_days):
        yield start_date + timedelta(n)


def get_article_details(pmid, retries=3, timeout=10):
    for _ in range(retries):
        try:
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'api_key': NCBI_API_KEY
            }
            response = requests.get(
                BASE_URL + 'efetch.fcgi', params=params, timeout=timeout)
            return response
        except requests.exceptions.Timeout:
            print("Request timeout, retrying...")
            time.sleep(1)
    print("Error: Failed to get article details after multiple retries.")
    return None


# Function to parse IDs from the NCBI API response
def parse_ids(response):
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        print("Error: Unable to parse the XML response.")
        print(response.content.decode())  # Added for debugging purposes
        return []

    id_list_element = root.find("IdList")
    if id_list_element is None:
        print("Error: Root element not found in the XML response.")
        print(response.content.decode())  # Added for debugging purposes
        return []

    pmids = [id_elem.text for id_elem in id_list_element.findall("Id")]
    return pmids


# parse article data from the NCBI API response
def parse_article_data(response):
    try:
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(response.content, parser=parser)
        article_data = []
        # Check if root is not None before calling findall
        if root is not None:
            article_data = []
            for article in root.findall('PubmedArticle'):
                pm_id = article.find('.//PMID').text
                title = article.find('.//ArticleTitle').text
                authors = [author.find('LastName').text + ' ' + author.find('Initials').text if author.find(
                    'LastName') is not None and author.find('Initials') is not None else '' for author in article.findall('.//Author')]
                abstract = ''.join(
                    [abstract_text.text if abstract_text.text else '' for abstract_text in article.findall('.//AbstractText')])
                article_data.append({
                    'pm_id': pm_id,
                    'title': title,
                    'authors': ', '.join(authors),
                    'abstract': abstract
                })
            return article_data
        else:
            # print("Error: Root element not found in the XML response")
            return []
    except etree.XMLSyntaxError as e:
        print("Error: Unable to parse the XML response in parse_article_data function")
        print(f"Error details: {str(e)}")
        print("Response content:")
        print(response.content)  # Print the problematic XML content
        return []


# Function to fetch and parse article data from a given PMID
def fetch_and_parse_article_data(pmid):
    article_response = get_article_details(pmid)
    article_data = parse_article_data(article_response)
    return article_data[0] if article_data else None


def scrape_pubmed_articles(query, retstart=0, num_threads=6, start_date=None, end_date=None, numArticles=5000):
    all_articles_data = []
    num = 0
    print(f"Number of threads: {num_threads}")
    max_articles = numArticles

    while retstart < max_articles:
        response = send_request(
            query, retstart, retmax=1000, mindate=start_date, maxdate=end_date)
        pmids = parse_ids(response)

        # Fetch and parse article data using multithreading
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_pmid = {executor.submit(
                fetch_and_parse_article_data, pmid): pmid for pmid in pmids}
            for future in concurrent.futures.as_completed(future_to_pmid):
                pmid = future_to_pmid[future]
                try:
                    article_data = future.result()
                    if article_data:
                        all_articles_data.append(article_data)
                        if num % 50 == 0:
                            print(num)
                        num += 1
                except Exception as exc:
                    print(f"{exc}")
                    print("hhh")

        retstart += 1000

    return all_articles_data


# Function to save scraped article data to a CSV file
def save_articles_to_csv(articles_data, filename=csvFile):
    fieldnames = ['pm_id', 'title', 'authors', 'abstract']

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles_data)


# Function to generate embeddings and upload them to Pinecone
def generate_embeddings_and_upload(inputFile=input_datapath, outputFile=output_embeddings_path):
    # Create embeddings using your provided createEmbeddings function.
    createEmbeddings(inputFile)

    # Initialize Pinecone and OpenAI.
    pinecone.init(PINECONE_API_KEY, environment="us-east-1-aws")
    openai.api_key = OPENAI_API_KEY

    indexName = "pm-embeddings-001"
    index = pinecone.Index(index_name=indexName)

    df = pd.read_csv(outputFile, usecols=[
                     "pm_id", "embedding", "combined", "authors"])
    df["embedding"] = df.embedding.apply(eval).apply(list)

    idEmbeddingPairs = list(zip(df['pm_id'].astype(str), df['embedding']))

    # Upload the embeddings to Pinecone.
    numChunks = 0
    for ids_vectors_chunk in chunks(idEmbeddingPairs, batch_size=100):
        print(numChunks)
        numChunks += 1
        index.upsert(vectors=ids_vectors_chunk, namespace="PM_ID")

    return len(idEmbeddingPairs)


def track_pubmed_statistics(query, date, num_sources, start, end):
    stats_file = 'Growth/PM_stats.json'
    date = str(date)
    start = str(start)
    end = str(end)

    # Read the existing statistics if the file exists
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except:
        stats = {}

    data = {"totalSources": 0, "query": {"query": query,
                                         "numSources": 0, "from": start, "to": end}}
    if date not in stats:
        stats[date] = data
    # if query not in stats[date]:
    #     stats[date][query] = {"numSources": 0, "article start dates": start, "end date": end}

    stats[date]["query"]["numSources"] += num_sources
    # stats[date]["query"]["from"] = end
    stats[date]["query"]["to"] = end
    stats[date]["totalSources"] += num_sources

    # Save the updated statistics to the file
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)


# Main function to scrape PubMed articles, save them as a CSV, and generate and upload embeddings to Pinecone
def main(start_from=None, start_date=None, end_date=None):
    currentDate = date.today().strftime("%m/%d/%Y")
    if start_date and end_date:
        start_date = date.fromisoformat(start_date)
        end_date = date.fromisoformat(end_date)

    query = "cardiology OR cardiovascular disease OR congestive heart failure OR CHF OR heart failure OR myocardial infarction OR MI OR STEMI OR NSTEMI OR pericardial disease OR endocarditis OR PCI OR percutaneous coronary intervention OR angiography OR PE OR pulmonary embolism OR CABG OR coronary artery bypass graft OR thrombectomy OR DVT OR deep vein thrombosis OR HFpEF OR HFrEF OR cardiac tamponade OR pericardial effusion OR amyloidosis OR systolic dysfunction OR diastolic dysfunction"
    max_articles = 9999  # Set the number of articles you want to scrape - gets ~775 articles per 1000 queried with 4 scraping threads

    print("\nScraping PubMed articles...\n")
    articles_data = scrape_pubmed_articles(
        query, start_date=start_date, end_date=end_date, numArticles=max_articles)
    print(f"Scraped {len(articles_data)} articles")

    print("Saving articles to CSV file...")
    save_articles_to_csv(articles_data)
    print("Done!")
    print("Generating embeddings and uploading to Pinecone...")
    numArticles = generate_embeddings_and_upload()
    print("Done!")
    track_pubmed_statistics(
        query, currentDate, numArticles, start_date, end_date)


def add_months(date_str, months):
    date_obj = date.fromisoformat(date_str)

    full_months = int(months)
    # Assuming 30 days in half a month
    remaining_days = int((months - full_months) * 30)

    month = date_obj.month - 1 + full_months
    year = date_obj.year + month // 12
    month = month % 12 + 1
    day = min(date_obj.day, [31, 29 if year % 4 == 0 and (
        year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])

    new_date = date_obj.replace(year=year, month=month, day=day)
    new_date += timedelta(days=remaining_days)

    return new_date.isoformat()
    

running = True
# Entry point for the script
if __name__ == '__main__':
    with open("notes/dates_queries", 'r') as f:
        dates = f.read()
    dates = dates.split(" -> ")
    start_date = dates[0]
    end_date = dates[1]
    interval_months = 0.5     # how many months in between PM queries
    numCycles = 0   # 24 / interval_months cycles in two years

    while running:
        print(f"{start_date} -> {end_date}")
        main(start_from=1, start_date=start_date, end_date=end_date)

        # Update start_date and end_date for the next iteration
        start_date = add_months(start_date, interval_months)
        end_date = add_months(end_date, interval_months)

        # keep track of most recent time range queried
        with open("notes/dates_queries", 'w') as f:
            f.write(start_date + " -> " + end_date)
            print("Start date saved!")

        # You can add a stopping condition if needed, for example, stop after a specific date
        if numCycles > 100:  # 100 / 12
            break
        numCycles += 1
 