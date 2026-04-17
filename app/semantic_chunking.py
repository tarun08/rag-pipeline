import sys
import os
from dotenv import load_dotenv

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from app.custom_model.gemini import GeminiPreviewEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


upi_text = """Unified Payments Interface (UPI) is an Indian instant payment system and protocol developed by the National 
    Payments Corporation of India (NPCI) in April 11th 2016. 
    The interface facilitates inter-bank peer-to-peer (P2P) and person-to-merchant (P2M) transactions.
    It is used on mobile devices to instantly transfer funds between two bank accounts using only a unique UPI ID. 
    It runs as an open source application programming interface (API) on top of the Immediate Payment Service (IMPS), and is 
    regulated by the Reserve Bank of India (RBI). Major Indian banks started making their UPI-enabled apps available to customers in August 2016
    and the system 
    is today supported by almost all Indian banks.On 1 August 2025, the NPCI implemented new technical and operational restrictions across
    UPI applications to enhance system stability and reduce the risk of fraud. 
    The changes include daily limits on balance enquiry and “list accounts” API calls, a regulated processing window for autopay 
    mandate scheduling, and caps.From 93,000 transactions in August 2016 valued at ₹ 3 crore (30 million), UPI generated 
    80 crore (800 million) transactions in March 2019 with a total value of ₹ 1.3 lakh crore (1330 billion).In June 2021, 
    UPI recorded 9.4 crore (94 million) IPO mandates that increased to 76.6 lakhs (7.66 million) in July.This was the highest 
    ever since UPI was made mandatory by Securities and Exchange Board of India for domestic retail investors for IPO process. 
    By late August 2020, with 18 billion annual transactions, UPI surpassed American Express in India. NITI Aayog predicted that UPI will 
    also surpass Visa and Mastercard by 2023. UPI took three years to reach 1.14 billion in October 2019 while by the end of October 2020,
    the payment system registered 2.07 billion transactions.In 2020, $457 billion worth of value was moved on UPI platform which was 15% of 
    India's GDP.UPI Wallet facility has been introduced for foreign tourists in India, Once set up, users can add funds using their preferred debit
    or credit card and start scanning to pay at over 20 million stores in India with no commission fee. 
    The wallet's wide acceptance means it is convenient for tourists to transact in any location, from roadside tea stalls to five-star resorts.
    There are apps for UPI wallet such as Cheq UPI.
    """

from langchain_experimental.text_splitter import SemanticChunker

embeddings = GeminiPreviewEmbeddings()
semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=7.0
)
chunks = semantic_chunker.split_text(upi_text)

print("-" * 100)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
print("-" * 100)