"""
Skrip EDA & Analisis Data Awal (CRISP-DM: Data Understanding)
Membaca file 'hasil_clustering.xlsx', menganalisis karakteristik teks,
mengecek distribusi program studi dan cluster awal, menyaring noise,
dan mengekspor dataset terstruktur ke folder data/raw/.
"""

import os
import sys
import zipfile
import xml.etree.ElementTree as ET
import csv
from collections import Counter
import re

# Tambahkan path root agar dapat import src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.preprocessing import clean_text, is_noise_text


def parse_xlsx_direct(file_path: str):
    """Membaca sheet XLSX secara langsung tanpa dependensi eksternal."""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    
    with zipfile.ZipFile(file_path, 'r') as z:
        shared_strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in tree.findall('m:si', ns):
                text_parts = [t.text for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t') if t.text]
                shared_strings.append(''.join(text_parts))

        wb = ET.fromstring(z.read('xl/workbook.xml'))
        sheet_map = {}
        for s in wb.findall('.//m:sheet', ns):
            name = s.attrib.get('name')
            r_id = s.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            sheet_map[r_id] = name

        wb_rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rel_map = {rel.attrib['Id']: rel.attrib['Target'] for rel in wb_rels}

        sheets_data = {}
        for r_id, name in sheet_map.items():
            raw_target = rel_map[r_id].lstrip('/')
            if not raw_target.startswith('xl/'):
                raw_target = 'xl/' + raw_target
            
            stree = ET.fromstring(z.read(raw_target))
            rows = stree.findall('.//m:row', ns)
            table = []
            for r in rows:
                cells = r.findall('m:c', ns)
                row_dict = {}
                for c in cells:
                    ref = c.attrib.get('r')
                    col = ''.join([ch for ch in ref if ch.isalpha()])
                    t_attr = c.attrib.get('t')
                    v = c.find('m:v', ns)
                    val = v.text if v is not None else ''
                    if t_attr == 's' and val.isdigit():
                        idx = int(val)
                        val = shared_strings[idx] if idx < len(shared_strings) else val
                    elif t_attr == 'inlineStr':
                        is_t = c.find('.//m:t', ns)
                        val = is_t.text if is_t is not None else ''
                    row_dict[col] = val
                table.append(row_dict)
            sheets_data[name] = table

        return sheets_data


def run_eda(excel_path: str = "hasil_clustering.xlsx"):
    print("=" * 70)
    print("📊 LAPORAN EKSPLORASI DATA AWAL (EDA) - SISTEM NLP TRACER STUDY")
    print("=" * 70)
    
    if not os.path.exists(excel_path):
        print(f"Error: File '{excel_path}' tidak ditemukan!")
        return

    sheets = parse_xlsx_direct(excel_path)
    os.makedirs(os.path.join(parent_dir, "data", "raw"), exist_ok=True)
    
    combined_records = []
    
    for sheet_name in ["Kritik", "Saran", "Testimoni"]:
        if sheet_name not in sheets:
            continue
            
        table = sheets[sheet_name]
        data_rows = table[1:] # Lewati header
        
        print(f"\n🔹 Sheet: [{sheet_name}] - Total Data Mentah: {len(data_rows)} baris")
        
        valid_rows = []
        noise_count = 0
        word_counts = []
        char_counts = []
        prodi_counter = Counter()
        cluster_counter = Counter()
        
        csv_out_path = os.path.join(parent_dir, "data", "raw", f"{sheet_name.lower()}_raw.csv")
        
        with open(csv_out_path, mode="w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["nim", "program_studi", "teks_asli", "teks_bersih", "cluster_id", "label_cluster", "is_noise"])
            
            for row in data_rows:
                nim = row.get("A", "").strip()
                prodi = row.get("B", "").strip()
                teks_asli = row.get("C", "").strip()
                cluster_id = row.get("E", "").strip()
                label_cluster = row.get("F", "").strip()
                
                noise = is_noise_text(teks_asli)
                cleaned = clean_text(teks_asli)
                
                if noise:
                    noise_count += 1
                else:
                    valid_rows.append(cleaned)
                    words = cleaned.split()
                    word_counts.append(len(words))
                    char_counts.append(len(cleaned))
                    prodi_counter[prodi] += 1
                    cluster_counter[f"Cluster {cluster_id} ({label_cluster})"] += 1
                    
                    # Tambahkan ke combined records
                    combined_records.append({
                        "kategori_sumber": sheet_name,
                        "nim": nim,
                        "program_studi": prodi,
                        "teks_asli": teks_asli,
                        "teks_bersih": cleaned,
                        "cluster_id": cluster_id,
                        "label_cluster": label_cluster
                    })
                    
                writer.writerow([nim, prodi, teks_asli, cleaned, cluster_id, label_cluster, 1 if noise else 0])
                
        print(f"   • Data Valid / Bermakna : {len(valid_rows)}")
        print(f"   • Data Noise Disaring   : {noise_count}")
        if word_counts:
            print(f"   • Statistik Kata/Teks   : Min={min(word_counts)}, Max={max(word_counts)}, Rata-rata={sum(word_counts)/len(word_counts):.2f}")
            print(f"   • Statistik Karakter    : Min={min(char_counts)}, Max={max(char_counts)}, Rata-rata={sum(char_counts)/len(char_counts):.2f}")
        print(f"   • Top 3 Program Studi   : {prodi_counter.most_common(3)}")
        print(f"   • Distribusi Cluster    : {dict(cluster_counter)}")
        print(f"   💾 Tersimpan ke         : {csv_out_path}")
        
    # Simpan dataset gabungan
    combined_csv_path = os.path.join(parent_dir, "data", "raw", "combined_tracer_study.csv")
    with open(combined_csv_path, mode="w", newline="", encoding="utf-8") as f_comb:
        writer = csv.DictWriter(f_comb, fieldnames=["kategori_sumber", "nim", "program_studi", "teks_asli", "teks_bersih", "cluster_id", "label_cluster"])
        writer.writeheader()
        writer.writerows(combined_records)
        
    print("\n" + "=" * 70)
    print(f"✅ EDA Selesai! Total data valid terakumulasi: {len(combined_records)} sampel.")
    print(f"💾 Dataset Gabungan tersimpan di: {combined_csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_eda()
