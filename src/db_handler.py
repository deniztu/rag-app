import os
import sqlite3
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel
from typing import List, Optional
from llm import initialize_language_model

# Load environment variables from a .env file
load_dotenv()

# Define Pydantic Classes
class PackagingInfo(BaseModel):
    product_code: Optional[str] = None
    product_description: Optional[str] = None
    packaging_unit: Optional[int] = None
    dimensions: Optional[str] = None  # Example: "25 mm x 23 mm x 70 mm"
    volume: Optional[float] = None  # dm³
    gross_weight: Optional[float] = None  # g

class ProductData(BaseModel):
    # General Product Information
    product_number: Optional[int] = None
    product_name: Optional[str] = None
    family_brand: Optional[str] = None
    ansi_code: Optional[str] = None
    global_order_reference: Optional[str] = None

    application_areas: List[str] = []  # Multiple areas
    product_features: List[str] = []  # List of benefits/features

    # Electrical Data
    nominal_wattage: Optional[int] = None
    nominal_voltage: Optional[float] = None

    # Photometric Data
    nominal_luminous_flux: Optional[int] = None
    useful_luminous_flux: Optional[int] = None
    use_value_refers_to_luminous_flux: Optional[int] = None
    illuminated_field: Optional[str] = None  # Example: "3.0*6.0 mm²"
    color_temperature: Optional[int] = None
    correlated_color_temperature: Optional[int] = None
    chromaticity_coordinate_x: Optional[float] = None
    chromaticity_coordinate_y: Optional[float] = None
    color_rendering_index: Optional[int] = None
    light_center_length: Optional[float] = None  # LCL in mm

    # Physical Attributes
    lamp_base: Optional[str] = None
    diameter_in: Optional[float] = None
    diameter_mm: Optional[float] = None
    length: Optional[float] = None
    product_weight: Optional[float] = None

    # Operating Conditions
    burning_position: Optional[str] = None
    dimmable: Optional[str] = None

    # Lifetime Data
    nominal_lifetime: Optional[int] = None

    # Regulatory Information
    primary_article_identifier: List[str] = []
    energy_efficiency_class: Optional[str] = None
    declaration_scip_database: Optional[str] = None
    candidate_list_substance_1: Optional[str] = None

    # Country Specific Information
    product_code: Optional[str] = None
    metel_code: Optional[str] = None
    seg_no: Optional[str] = None
    stk_number: Optional[str] = None
    uk_org: Optional[str] = None

    # Packaging Information
    packaging_info: List[PackagingInfo] = []  # Stores multiple packaging formats

# Initialize the LLM
llm = initialize_language_model()
struc_llm = llm.with_structured_output(ProductData, method="function_calling")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text = "\n".join([doc.page_content for doc in documents])
    return text

def save_to_db(conn, extracted_data, file_path):
    cursor = conn.cursor()

    # Insert into surgical_lighting
    product = extracted_data  # JSON from LLM
    cursor.execute("""
        INSERT INTO surgical_lighting (file_path,
            product_number, product_name, family_brand, ansi_code, global_order_reference,
            nominal_wattage, nominal_voltage, nominal_luminous_flux, useful_luminous_flux,
            use_value_refers_to_luminous_flux, illuminated_field, color_temperature,
            correlated_color_temperature, chromaticity_coordinate_x, chromaticity_coordinate_y,
            color_rendering_index, light_center_length, lamp_base, diameter_in, diameter_mm,
            length, product_weight, burning_position, dimmable, nominal_lifetime,
            energy_efficiency_class, declaration_scip_database, candidate_list_substance_1,
            product_code, metel_code, seg_no, stk_number, uk_org
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_path,
        product.get("product_number"), product.get("product_name"), product.get("family_brand"), product.get("ansi_code"),
        product.get("global_order_reference"), product.get("nominal_wattage"), product.get("nominal_voltage"),
        product.get("nominal_luminous_flux"), product.get("useful_luminous_flux"), product.get("use_value_refers_to_luminous_flux"),
        product.get("illuminated_field"), product.get("color_temperature"), product.get("correlated_color_temperature"),
        product.get("chromaticity_coordinate_x"), product.get("chromaticity_coordinate_y"), product.get("color_rendering_index"),
        product.get("light_center_length"), product.get("lamp_base"), product.get("diameter_in"), product.get("diameter_mm"),
        product.get("length"), product.get("product_weight"), product.get("burning_position"), product.get("dimmable"),
        product.get("nominal_lifetime"), product.get("energy_efficiency_class"), product.get("declaration_scip_database"),
        product.get("candidate_list_substance_1"), product.get("product_code"), product.get("metel_code"),
        product.get("seg_no"), product.get("stk_number"), product.get("uk_org")
    ))

    # Get the inserted product ID
    product_id = cursor.lastrowid

    # Insert primary article identifiers
    primary_article_identifiers = product.get("primary_article_identifier", [])
    for identifier in primary_article_identifiers:
        cursor.execute("INSERT INTO primary_article_identifiers (product_id, identifier) VALUES (?, ?)", (product_id, identifier))

    # Insert application areas
    application_areas = product.get("application_areas", [])
    for area in application_areas:
        cursor.execute("INSERT INTO application_areas (product_id, application_area) VALUES (?, ?)", (product_id, area))

    # Insert product features
    product_features = product.get("product_features", [])
    for feature in product_features:
        cursor.execute("INSERT INTO product_features (product_id, feature) VALUES (?, ?)", (product_id, feature))

    # Insert packaging info
    packaging_info = product.get("packaging_info", [])
    for pack in packaging_info:
        cursor.execute("""
            INSERT INTO packaging_info (product_id, product_code, product_description, packaging_unit, dimensions, volume, gross_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (product_id, pack.get("product_code"), pack.get("product_description"), pack.get("packaging_unit"), pack.get("dimensions"), pack.get("volume"), pack.get("gross_weight")))

    # Commit the transaction
    conn.commit()

def process_pdfs_to_db(data_path, db_path):
    """
    Process PDF files in the specified directory and save the extracted data to the SQLite database.

    Args:
        data_path (str): The path to the directory containing PDF files.
        db_path (str): The path to the SQLite database file.
    """

    # Connect to SQLite
    conn = sqlite3.connect("lighting.db")
    cursor = conn.cursor()

    # Create tables if they do not exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS surgical_lighting (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_path TEXT,
        product_number INTEGER,
        product_name TEXT,
        family_brand TEXT,
        ansi_code TEXT,
        global_order_reference TEXT,
        nominal_wattage INTEGER,
        nominal_voltage REAL,
        nominal_luminous_flux INTEGER,
        useful_luminous_flux INTEGER,
        use_value_refers_to_luminous_flux INTEGER,
        illuminated_field TEXT,
        color_temperature INTEGER,
        correlated_color_temperature INTEGER,
        chromaticity_coordinate_x REAL,
        chromaticity_coordinate_y REAL,
        color_rendering_index INTEGER,
        light_center_length REAL,
        lamp_base TEXT,
        diameter_in REAL,
        diameter_mm REAL,
        length REAL,
        product_weight REAL,
        burning_position TEXT,
        dimmable TEXT,
        nominal_lifetime INTEGER,
        energy_efficiency_class TEXT,
        declaration_scip_database TEXT,
        candidate_list_substance_1 TEXT,
        product_code TEXT,
        metel_code TEXT,
        seg_no TEXT,
        stk_number TEXT,
        uk_org TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS primary_article_identifiers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        identifier TEXT,
        FOREIGN KEY (product_id) REFERENCES surgical_lighting (id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS application_areas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        application_area TEXT,
        FOREIGN KEY (product_id) REFERENCES surgical_lighting (id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        feature TEXT,
        FOREIGN KEY (product_id) REFERENCES surgical_lighting (id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS packaging_info (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER,
        product_code TEXT,
        product_description TEXT,
        packaging_unit INTEGER,
        dimensions TEXT,
        volume REAL,
        gross_weight REAL,
        FOREIGN KEY (product_id) REFERENCES surgical_lighting (id)
    )
    """)

    # Commit & Close
    conn.commit()
    conn.close()

    print("Tables created successfully.")


    # Connect to SQLite
    conn = sqlite3.connect(db_path, timeout=40.0)

    # Process each PDF file in the directory
    for filename in os.listdir(data_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_path, filename)
            text = extract_text_from_pdf(file_path)

            prompt = f"Extract structured data from the following text:\n\n{text}."
            
            # Assuming struc_llm.invoke(prompt) returns a structured JSON string
            result = struc_llm.invoke(prompt)
            extracted_data = json.loads(result.model_dump_json())
            
            # Save the extracted data to the database
            save_to_db(conn, extracted_data, file_path)
            print(f"Saved data from '{filename}' to the database.")

    # Close the connection
    conn.close()

# # Example usage
# if __name__ == "__main__":
#     DATA_PATH = "../data"
#     DB_PATH = "lighting.db"
#     process_pdfs_to_db(DATA_PATH, DB_PATH)