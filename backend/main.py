# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# import pandas as pd
# import io
# import os
# from typing import List, Dict, Tuple, Optional
# from datetime import datetime
# import traceback
# import tempfile

# app = FastAPI(title="Insurance Payout Processor API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# FORMULA_DATA = [
#     {"LOB": "TW", "SEGMENT": "1+5",            "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW TP"         ,  "PO": "-2%",          "REMARKS":"Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP"         ,  "PO": "-3%",          "REMARKS":"Payin  21% to 30%"},
#     {"LOB": "TW", "SEGMENT": "TW TP"         ,  "PO": "-4%",          "REMARKS":"Payin  31% to 50%"},
#     {"LOB": "TW", "SEGMENT": "TW TP"         ,  "PO": "-5%",          "REMARKS":"Payin Above 50%"},
# ]

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# import pandas as pd
# import io
# import os
# from typing import List, Dict, Tuple, Optional
# from datetime import datetime
# import traceback
# import tempfile

# app = FastAPI(title="Two Wheelers Payout Processor API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ===============================================================================
# # FORMULA DATA
# # ===============================================================================
# FORMULA_DATA = [
#     {"LOB": "TW", "SEGMENT": "1+5",            "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "PO": "90% of Payin", "REMARKS": "NIL"},
#     {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-2%",          "REMARKS": "Payin Below 20%"},
#     {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-3%",          "REMARKS": "Payin 21% to 30%"},
#     {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-4%",          "REMARKS": "Payin 31% to 50%"},
#     {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-5%",          "REMARKS": "Payin Above 50%"},
# ]

# # ===============================================================================
# # STATE MAPPING
# # ===============================================================================
# STATE_MAPPING = {
#     "ANDHRA PRADESH": "ANDHRA PRADESH",
#     "KRISHNA": "ANDHRA PRADESH",
#     "VIJAYWADA": "ANDHRA PRADESH",
#     "VIJAYAWADA": "ANDHRA PRADESH",
#     "VISAKHAPATNAM": "ANDHRA PRADESH",
#     "VIZAG": "ANDHRA PRADESH",
    
#     "KARNATAKA": "KARNATAKA",
#     "BANGALORE": "KARNATAKA",
#     "BENGALURU": "KARNATAKA",
    
#     "KERALA": "KERALA",
#     "ERNAKULAM": "KERALA",
#     "COCHIN": "KERALA",
    
#     "TAMIL NADU": "TAMIL NADU",
#     "CHENNAI": "TAMIL NADU",
#     "PONDICHERRY": "TAMIL NADU",
    
#     "TELANGANA": "TELANGANA",
#     "HYDERABAD": "TELANGANA",
    
#     "MAHARASHTRA": "MAHARASHTRA",
#     "MUMBAI": "MAHARASHTRA",
#     "PUNE": "MAHARASHTRA",
#     "NAGPUR": "MAHARASHTRA",
    
#     "MADHYA PRADESH": "MADHYA PRADESH",
#     "BHOPAL": "MADHYA PRADESH",
#     "INDORE": "MADHYA PRADESH",
#     "GWALIOR": "MADHYA PRADESH",
#     "JABALPUR": "MADHYA PRADESH",
    
#     "GOA": "GOA",
# }

# uploaded_files = {}

# # ===============================================================================
# # HELPER FUNCTIONS
# # ===============================================================================

# def cell_to_str(val) -> str:
#     """Safely convert ANY cell value to string."""
#     if val is None:
#         return ""
#     try:
#         if pd.isna(val):
#             return ""
#     except (TypeError, ValueError):
#         pass
#     return str(val).strip()


# def safe_float(value) -> Optional[float]:
#     """Safely convert value to float, handling percentages."""
#     if value is None:
#         return None
#     try:
#         if pd.isna(value):
#             return None
#     except (TypeError, ValueError):
#         pass
    
#     s = str(value).strip().upper().replace("%", "")
#     if s in ["D", "NA", "", "NAN", "NONE", "DECLINE", "0.00%", "0.0%", "0%"]:
#         return None
    
#     try:
#         num = float(s)
#         if num < 0:
#             return None
#         # Convert decimals to percentages (0.28 -> 28%)
#         return num * 100 if 0 < num < 1 else num
#     except Exception:
#         return None


# def map_state(location: str) -> str:
#     """Map location to state."""
#     location_upper = location.upper()
    
#     for key, val in STATE_MAPPING.items():
#         if key.upper() in location_upper:
#             return val
    
#     return location


# def get_payin_category(payin: float) -> str:
#     """Get payin category for TP."""
#     if payin <= 20:
#         return "Payin Below 20%"
#     elif payin <= 30:
#         return "Payin 21% to 30%"
#     elif payin <= 50:
#         return "Payin 31% to 50%"
#     else:
#         return "Payin Above 50%"


# def calculate_payout(payin: float, policy_type: str, lob: str = "TW", segment: str = "TW SAOD + COMP") -> Tuple[float, str, str]:
#     """
#     Calculate payout based on policy type.
#     CRITICAL: If payin <= 5%, return payout = 0
#     """
#     if payin is None or payin == 0:
#         return 0, "0% (No Payin)", "Payin is 0"
    
#     # CRITICAL: If payin <= 5%, set payout to 0
#     if payin <= 5:
#         return 0, "0% (Payin ≤ 5%)", f"Payin is {payin}% which is ≤ 5%, so payout = 0"
    
#     # For COMP and SAOD: 90% of Payin
#     if policy_type in ["COMP", "SAOD"]:
#         payout = round(payin * 0.90, 2)
#         formula = "90% of Payin"
#         explanation = f"Applied formula: {formula} for {policy_type}"
#         return payout, formula, explanation
    
#     # For TP: deduction based on payin range
#     elif policy_type == "TP":
#         payin_cat = get_payin_category(payin)
        
#         if payin <= 20:
#             deduction = 2
#         elif payin <= 30:
#             deduction = 3
#         elif payin <= 50:
#             deduction = 4
#         else:
#             deduction = 5
        
#         payout = round(payin - deduction, 2)
#         formula = f"-{deduction}%"
#         explanation = f"Applied formula: {formula} for TP, {payin_cat}"
#         return payout, formula, explanation
    
#     # Default fallback
#     return 0, "Unknown", "Unknown policy type"


# # ===============================================================================
# # PATTERN DETECTION
# # ===============================================================================

# class TWPatternDetector:
#     """Detect TW pattern type."""
    
#     @staticmethod
#     def detect_pattern(df: pd.DataFrame) -> str:
#         """
#         Detect pattern:
#         - 'tw_comp': TW COMP pattern (Geo Locations | Type | Payout % - Net | SOD)
#         - 'tw_satp': TW SATP pattern (Segment | Geo Location - New | CC bands)
#         """
#         sample_text = ""
#         for i in range(min(10, df.shape[0])):
#             row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
#             sample_text += row_text + " "
        
#         # Check for TW SATP
#         if ("SATP" in sample_text or "SA TP" in sample_text) and \
#            ("BIKES" in sample_text or "SCOOTER" in sample_text) and \
#            ("CC" in sample_text or "75" in sample_text):
#             return "tw_satp"
        
#         # Check for TW COMP
#         if ("TW COMP" in sample_text or "TW_COMP" in sample_text or "PAYOUT % - NET" in sample_text) and \
#            ("GEO LOCATION" in sample_text or "TYPE" in sample_text):
#             return "tw_comp"
        
#         # Default
#         return "tw_comp"


# # ===============================================================================
# # TW COMP PROCESSOR
# # ===============================================================================

# class TWCompProcessor:
#     """Process TW COMP sheets."""
    
#     @staticmethod
#     def process(content: bytes, sheet_name: str,
#                 override_enabled: bool = False,
#                 override_lob: str = None,
#                 override_segment: str = None) -> List[Dict]:
#         """
#         Process TW COMP pattern:
#         Row 1: Title (JAN 2025 PAYOUT - TW COMP)
#         Row 2: Geo Locations | Type (Package 1+1) | Payout % - Net | SOD
#         Row 4+: Data rows
#         """
#         records = []
        
#         try:
#             df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
            
#             print(f"\n[TW_COMP] Processing sheet: {sheet_name}")
#             print(f"[TW_COMP] Sheet shape: {df.shape}")
            
#             # Find header row
#             header_row = None
#             for i in range(min(10, df.shape[0])):
#                 row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
#                 if "GEO LOCATION" in row_text and ("TYPE" in row_text or "PACKAGE" in row_text):
#                     header_row = i
#                     break
            
#             if header_row is None:
#                 print("[TW_COMP] Header row not found")
#                 return records
            
#             print(f"[TW_COMP] Found header row at index: {header_row}")
            
#             # Data starts after header + skip empty row
#             data_start = header_row + 1
#             for i in range(data_start, df.shape[0]):
#                 if cell_to_str(df.iloc[i, 0]):
#                     data_start = i
#                     break
            
#             # Build column metadata
#             col_meta = []
#             for col_idx in range(2, df.shape[1]):
#                 header = cell_to_str(df.iloc[header_row, col_idx]).upper()
                
#                 if not header:
#                     continue
                
#                 # Determine policy type
#                 if "PAYOUT" in header and "NET" in header:
#                     policy_type = "COMP"
#                 elif "SOD" in header or "SAOD" in header:
#                     policy_type = "SAOD"
#                 else:
#                     continue
                
#                 col_meta.append({
#                     "col_idx": col_idx,
#                     "header": cell_to_str(df.iloc[header_row, col_idx]),
#                     "policy_type": policy_type,
#                 })
            
#             if not col_meta:
#                 print("[TW_COMP] No data columns found")
#                 return records
            
#             print(f"[TW_COMP] Found {len(col_meta)} columns")
            
#             # Process data rows
#             lob_final = override_lob if override_enabled and override_lob else "TW"
#             segment_final = override_segment if override_enabled and override_segment else "TW SAOD + COMP"
            
#             skip_words = {"total", "grand total", "average", "sum", ""}
            
#             for row_idx in range(data_start, df.shape[0]):
#                 geo_location = cell_to_str(df.iloc[row_idx, 0])
#                 tw_type = cell_to_str(df.iloc[row_idx, 1])  # Bike or Scooter
                
#                 if not geo_location or geo_location.lower() in skip_words:
#                     continue
                
#                 state = map_state(geo_location)
                
#                 # Process each column
#                 for m in col_meta:
#                     payin = safe_float(df.iloc[row_idx, m["col_idx"]])
                    
#                     if payin is None:
#                         continue
                    
#                     payout, formula, explanation = calculate_payout(payin, m["policy_type"], lob_final, segment_final)
                    
#                     records.append({
#                         "State": state,
#                         "Geo Location": geo_location,
#                         "TW Type": tw_type,  # Bike or Scooter
#                         "Original Segment": m["header"],
#                         "Mapped Segment": segment_final,
#                         "LOB": lob_final,
#                         "Policy Type": m["policy_type"],
#                         "Status": "STP",
#                         "Payin": f"{payin:.2f}%",
#                         "Calculated Payout": f"{payout:.2f}%",
#                         "Formula Used": formula,
#                         "Rule Explanation": explanation,
#                     })
            
#             print(f"[TW_COMP] Extracted {len(records)} records")
#             return records
            
#         except Exception as e:
#             print(f"[TW_COMP] Error: {e}")
#             traceback.print_exc()
#             return []


# # ===============================================================================
# # TW SATP PROCESSOR
# # ===============================================================================

# class TWSATPProcessor:
#     """Process TW SATP sheets."""
    
#     @staticmethod
#     def process(content: bytes, sheet_name: str,
#                 override_enabled: bool = False,
#                 override_lob: str = None,
#                 override_segment: str = None) -> List[Dict]:
#         """
#         Process TW SATP pattern:
#         Row 1: Title (JAN 2025 PAYOUT_TW SATP)
#         Row 2: Segment (Bikes | Scooter)
#         Row 3: Geo Location - New | CC bands
#         Row 4+: Data rows
#         """
#         records = []
        
#         try:
#             df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
            
#             print(f"\n[TW_SATP] Processing sheet: {sheet_name}")
#             print(f"[TW_SATP] Sheet shape: {df.shape}")
            
#             # Find segment row (Bikes/Scooter)
#             segment_row = None
#             for i in range(min(10, df.shape[0])):
#                 row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
#                 if "BIKES" in row_text or "SCOOTER" in row_text:
#                     segment_row = i
#                     break
            
#             if segment_row is None:
#                 print("[TW_SATP] Segment row not found")
#                 return records
            
#             print(f"[TW_SATP] Found segment row at index: {segment_row}")
            
#             # Next row is Geo Location row
#             geo_header_row = segment_row + 1
            
#             # Data starts after geo header
#             data_start = geo_header_row + 1
#             for i in range(data_start, df.shape[0]):
#                 if cell_to_str(df.iloc[i, 0]):
#                     data_start = i
#                     break
            
#             # Build column metadata
#             col_meta = []
#             last_segment = ""
            
#             for col_idx in range(1, df.shape[1]):
#                 segment = cell_to_str(df.iloc[segment_row, col_idx]).upper()
#                 cc_band = cell_to_str(df.iloc[geo_header_row, col_idx])
                
#                 if not segment and not cc_band:
#                     continue
                
#                 # Forward fill segment (for merged cells)
#                 if segment and ("BIKES" in segment or "SCOOTER" in segment):
#                     last_segment = segment
                
#                 if not last_segment:
#                     continue
                
#                 # Build description
#                 tw_type = "Bikes" if "BIKES" in last_segment else "Scooter"
#                 segment_desc = f"{tw_type}"
#                 if cc_band:
#                     segment_desc += f" ({cc_band})"
                
#                 col_meta.append({
#                     "col_idx": col_idx,
#                     "tw_type": tw_type,
#                     "cc_band": cc_band,
#                     "segment_desc": segment_desc,
#                 })
            
#             if not col_meta:
#                 print("[TW_SATP] No data columns found")
#                 return records
            
#             print(f"[TW_SATP] Found {len(col_meta)} columns")
            
#             # Process data rows
#             lob_final = override_lob if override_enabled and override_lob else "TW"
#             segment_final = override_segment if override_enabled and override_segment else "TW TP"
            
#             skip_words = {"total", "grand total", "average", "sum", ""}
            
#             for row_idx in range(data_start, df.shape[0]):
#                 geo_location = cell_to_str(df.iloc[row_idx, 0])
                
#                 if not geo_location or geo_location.lower() in skip_words:
#                     continue
                
#                 state = map_state(geo_location)
                
#                 # Process each column
#                 for m in col_meta:
#                     payin = safe_float(df.iloc[row_idx, m["col_idx"]])
                    
#                     if payin is None:
#                         continue
                    
#                     payout, formula, explanation = calculate_payout(payin, "TP", lob_final, segment_final)
                    
#                     records.append({
#                         "State": state,
#                         "Geo Location": geo_location,
#                         "TW Type": m["tw_type"],  # Bikes or Scooter
#                         "Original Segment": m["segment_desc"],
#                         "Mapped Segment": segment_final,
#                         "LOB": lob_final,
#                         "Policy Type": "TP",
#                         "CC Band": m["cc_band"],
#                         "Status": "STP",
#                         "Payin": f"{payin:.2f}%",
#                         "Calculated Payout": f"{payout:.2f}%",
#                         "Formula Used": formula,
#                         "Rule Explanation": explanation,
#                     })
            
#             print(f"[TW_SATP] Extracted {len(records)} records")
#             return records
            
#         except Exception as e:
#             print(f"[TW_SATP] Error: {e}")
#             traceback.print_exc()
#             return []


# # ===============================================================================
# # PATTERN DISPATCHER
# # ===============================================================================

# class TWPatternDispatcher:
#     """Route to correct TW processor."""
    
#     PATTERN_PROCESSORS = {
#         "tw_comp": TWCompProcessor,
#         "tw_satp": TWSATPProcessor,
#     }
    
#     @staticmethod
#     def process_sheet(content: bytes, sheet_name: str,
#                       override_enabled: bool = False,
#                       override_lob: str = None,
#                       override_segment: str = None) -> List[Dict]:
#         """Detect pattern and route to processor."""
#         df_raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
#         pattern = TWPatternDetector.detect_pattern(df_raw)
        
#         print(f"\n[DISPATCHER] Detected pattern: {pattern}")
        
#         processor_class = TWPatternDispatcher.PATTERN_PROCESSORS.get(pattern, TWCompProcessor)
#         return processor_class.process(
#             content, sheet_name,
#             override_enabled, override_lob, override_segment
#         )


# # ===============================================================================
# # API ENDPOINTS
# # ===============================================================================

# @app.get("/")
# async def root():
#     return {
#         "message": "Two Wheelers Payout Processor API",
#         "version": "1.0.0",
#         "formula": "90% for COMP/SAOD, Tiered deduction for TP",
#         "supported_lobs": ["TW"],
#         "supported_segments": ["TW SAOD + COMP", "TW TP"],
#         "supported_patterns": [
#             "tw_comp - TW COMP (Geo Locations | Type | Payout % - Net | SOD)",
#             "tw_satp - TW SATP (Segment | Geo Location | CC bands for Bikes/Scooter)"
#         ],
#         "special_rules": [
#             "If payin ≤ 5%, payout = 0"
#         ]
#     }


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     """Upload Excel file."""
#     try:
#         if not file.filename.endswith((".xlsx", ".xls")):
#             raise HTTPException(status_code=400, detail="Only Excel files supported")
        
#         content = await file.read()
#         xls = pd.ExcelFile(io.BytesIO(content))
#         sheets = xls.sheet_names
        
#         file_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         uploaded_files[file_id] = {
#             "content": content,
#             "filename": file.filename,
#             "sheets": sheets,
#         }
        
#         return {
#             "file_id": file_id,
#             "filename": file.filename,
#             "sheets": sheets,
#             "message": f"Uploaded successfully. {len(sheets)} worksheet(s) found.",
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


# @app.post("/process")
# async def process_sheet(
#     file_id: str,
#     sheet_name: str,
#     override_enabled: bool = False,
#     override_lob: Optional[str] = None,
#     override_segment: Optional[str] = None,
# ):
#     """Process worksheet."""
#     try:
#         if file_id not in uploaded_files:
#             raise HTTPException(status_code=404, detail="File not found")
        
#         file_data = uploaded_files[file_id]
        
#         if sheet_name not in file_data["sheets"]:
#             raise HTTPException(status_code=400, detail=f"Sheet '{sheet_name}' not found")
        
#         records = TWPatternDispatcher.process_sheet(
#             file_data["content"], 
#             sheet_name,
#             override_enabled, 
#             override_lob, 
#             override_segment,
#         )
        
#         if not records:
#             return {
#                 "success": False,
#                 "message": "No records extracted. Check sheet structure.",
#                 "records": [],
#                 "count": 0,
#             }
        
#         # Summary stats
#         states = {}
#         policies = {}
#         payins = []
#         payouts = []
        
#         for r in records:
#             state = r.get("State", "UNKNOWN")
#             states[state] = states.get(state, 0) + 1
            
#             policy = r.get("Policy Type", "UNKNOWN")
#             policies[policy] = policies.get(policy, 0) + 1
            
#             try:
#                 payin_val = float(r.get("Payin", "0%").replace("%", ""))
#                 payout_val = float(r.get("Calculated Payout", "0%").replace("%", ""))
#                 payins.append(payin_val)
#                 payouts.append(payout_val)
#             except Exception:
#                 pass
        
#         avg_payin = round(sum(payins) / len(payins), 2) if payins else 0
#         avg_payout = round(sum(payouts) / len(payouts), 2) if payouts else 0
        
#         return {
#             "success": True,
#             "message": f"Successfully processed {len(records)} records from '{sheet_name}'",
#             "records": records,
#             "count": len(records),
#             "summary": {
#                 "total_records": len(records),
#                 "states": dict(sorted(states.items(), key=lambda x: x[1], reverse=True)[:10]),
#                 "policy_types": policies,
#                 "average_payin": avg_payin,
#                 "average_payout": avg_payout,
#             },
#         }
        
#     except Exception as e:
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# @app.post("/export")
# async def export_to_excel(file_id: str, sheet_name: str, records: List[Dict]):
#     """Export to Excel."""
#     try:
#         if not records:
#             raise HTTPException(status_code=400, detail="No records to export")
        
#         df = pd.DataFrame(records)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"TW_Processed_{sheet_name.replace(' ', '_')}_{timestamp}.xlsx"
#         out_path = os.path.join(tempfile.gettempdir(), filename)
        
#         with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
#             df.to_excel(writer, index=False, sheet_name="Processed Data")
            
#             worksheet = writer.sheets["Processed Data"]
#             for idx, col in enumerate(df.columns):
#                 max_length = max(
#                     df[col].astype(str).apply(len).max(),
#                     len(str(col))
#                 )
#                 worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
#         return FileResponse(
#             path=out_path,
#             filename=filename,
#             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


# @app.get("/health")
# async def health_check():
#     """Health check."""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "uploaded_files": len(uploaded_files)
#     }


# if __name__ == "__main__":
#     import uvicorn
#     print("\n" + "=" * 70)
#     print("Two Wheelers Payout Processor API - v1.0.0")
#     print("Patterns: TW COMP + TW SATP")
#     print("Special Rule: Payin ≤ 5% → Payout = 0")
#     print("=" * 70 + "\n")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import io
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import traceback
import tempfile

app = FastAPI(title="Two Wheelers Payout Processor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================================================================
# FORMULA DATA
# ===============================================================================
FORMULA_DATA = [
    {"LOB": "TW", "SEGMENT": "1+5",            "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW SAOD + COMP", "PO": "90% of Payin", "REMARKS": "NIL"},
    {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-2%",          "REMARKS": "Payin Below 20%"},
    {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-3%",          "REMARKS": "Payin 21% to 30%"},
    {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-4%",          "REMARKS": "Payin 31% to 50%"},
    {"LOB": "TW", "SEGMENT": "TW TP",          "PO": "-5%",          "REMARKS": "Payin Above 50%"},
]

# ===============================================================================
# STATE MAPPING
# ===============================================================================
STATE_MAPPING = {
    "ANDHRA PRADESH": "ANDHRA PRADESH",
    "KRISHNA": "ANDHRA PRADESH",
    "VIJAYWADA": "ANDHRA PRADESH",
    "VIJAYAWADA": "ANDHRA PRADESH",
    "VISAKHAPATNAM": "ANDHRA PRADESH",
    "VIZAG": "ANDHRA PRADESH",
    
    "KARNATAKA": "KARNATAKA",
    "BANGALORE": "KARNATAKA",
    "BENGALURU": "KARNATAKA",
    
    "KERALA": "KERALA",
    "ERNAKULAM": "KERALA",
    "COCHIN": "KERALA",
    
    "TAMIL NADU": "TAMIL NADU",
    "CHENNAI": "TAMIL NADU",
    "PONDICHERRY": "TAMIL NADU",
    
    "TELANGANA": "TELANGANA",
    "HYDERABAD": "TELANGANA",
    
    "MAHARASHTRA": "MAHARASHTRA",
    "MUMBAI": "MAHARASHTRA",
    "PUNE": "MAHARASHTRA",
    "NAGPUR": "MAHARASHTRA",
    
    "MADHYA PRADESH": "MADHYA PRADESH",
    "BHOPAL": "MADHYA PRADESH",
    "INDORE": "MADHYA PRADESH",
    "GWALIOR": "MADHYA PRADESH",
    "JABALPUR": "MADHYA PRADESH",
    
    "GOA": "GOA",
}

uploaded_files = {}

# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def cell_to_str(val) -> str:
    """Safely convert ANY cell value to string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip()


def safe_float(value) -> Optional[float]:
    """Safely convert value to float, handling percentages."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    
    s = str(value).strip().upper().replace("%", "")
    if s in ["D", "NA", "", "NAN", "NONE", "DECLINE", "0.00%", "0.0%", "0%"]:
        return None
    
    try:
        num = float(s)
        if num < 0:
            return None
        # Convert decimals to percentages (0.28 -> 28%)
        return num * 100 if 0 < num < 1 else num
    except Exception:
        return None


def map_state(location: str) -> str:
    """Map location to state."""
    location_upper = location.upper()
    
    for key, val in STATE_MAPPING.items():
        if key.upper() in location_upper:
            return val
    
    return location


def get_payin_category(payin: float) -> str:
    """Get payin category for TP."""
    if payin <= 20:
        return "Payin Below 20%"
    elif payin <= 30:
        return "Payin 21% to 30%"
    elif payin <= 50:
        return "Payin 31% to 50%"
    else:
        return "Payin Above 50%"


def calculate_payout(payin: float, policy_type: str, lob: str = "TW", segment: str = "TW SAOD + COMP") -> Tuple[float, str, str]:
    """
    Calculate payout based on policy type.
    CRITICAL: If payin <= 5%, return payout = 0
    """
    if payin is None or payin == 0:
        return 0, "0% (No Payin)", "Payin is 0"
    
    # CRITICAL: If payin <= 5%, set payout to 0
    if payin <= 5:
        return 0, "0% (Payin ≤ 5%)", f"Payin is {payin}% which is ≤ 5%, so payout = 0"
    
    # For COMP and SAOD: 90% of Payin
    if policy_type in ["COMP", "SAOD"]:
        payout = round(payin * 0.90, 2)
        formula = "90% of Payin"
        explanation = f"Applied formula: {formula} for {policy_type}"
        return payout, formula, explanation
    
    # For TP: deduction based on payin range
    elif policy_type == "TP":
        payin_cat = get_payin_category(payin)
        
        if payin <= 20:
            deduction = 2
        elif payin <= 30:
            deduction = 3
        elif payin <= 50:
            deduction = 4
        else:
            deduction = 5
        
        payout = round(payin - deduction, 2)
        formula = f"-{deduction}%"
        explanation = f"Applied formula: {formula} for TP, {payin_cat}"
        return payout, formula, explanation
    
    # Default fallback
    return 0, "Unknown", "Unknown policy type"


# ===============================================================================
# PATTERN DETECTION
# ===============================================================================

class TWPatternDetector:
    """Detect TW pattern type."""
    
    @staticmethod
    def detect_pattern(df: pd.DataFrame) -> str:
        """
        Detect pattern:
        - 'tw_comp': TW COMP pattern (Geo Locations | Type | Payout % - Net | SOD)
        - 'tw_satp': TW SATP pattern (Segment | Geo Location - New | CC bands)
        - 'tw_satp_geo_new_old': TW SATP with Geo segment New/Old columns
        """
        sample_text = ""
        for i in range(min(10, df.shape[0])):
            row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
            sample_text += row_text + " "
        
        # Check for TW SATP with Geo New/Old
        has_geo_new = "GEO SEGMENT NEW" in sample_text or "GEO SEGMENT OLD" in sample_text
        has_satp = "SATP" in sample_text or "TWSATP" in sample_text
        has_bikes_scooter = "BIKES" in sample_text or "SCOOTER" in sample_text
        
        if has_geo_new and has_satp and has_bikes_scooter:
            return "tw_satp_geo_new_old"
        
        # Check for regular TW SATP
        if (has_satp or "SA TP" in sample_text) and \
           has_bikes_scooter and \
           ("CC" in sample_text or "75" in sample_text):
            return "tw_satp"
        
        # Check for TW COMP
        if ("TW COMP" in sample_text or "TW_COMP" in sample_text or "PAYOUT % - NET" in sample_text) and \
           ("GEO LOCATION" in sample_text or "TYPE" in sample_text):
            return "tw_comp"
        
        # Default
        return "tw_comp"


# ===============================================================================
# TW COMP PROCESSOR
# ===============================================================================

class TWCompProcessor:
    """Process TW COMP sheets."""
    
    @staticmethod
    def process(content: bytes, sheet_name: str,
                override_enabled: bool = False,
                override_lob: str = None,
                override_segment: str = None) -> List[Dict]:
        """
        Process TW COMP pattern:
        Row 1: Title (JAN 2025 PAYOUT - TW COMP)
        Row 2: Geo Locations | Type (Package 1+1) | Payout % - Net | SOD
        Row 4+: Data rows
        """
        records = []
        
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
            
            print(f"\n[TW_COMP] Processing sheet: {sheet_name}")
            print(f"[TW_COMP] Sheet shape: {df.shape}")
            
            # Find header row
            header_row = None
            for i in range(min(10, df.shape[0])):
                row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
                if "GEO LOCATION" in row_text and ("TYPE" in row_text or "PACKAGE" in row_text):
                    header_row = i
                    break
            
            if header_row is None:
                print("[TW_COMP] Header row not found")
                return records
            
            print(f"[TW_COMP] Found header row at index: {header_row}")
            
            # Data starts after header + skip empty row
            data_start = header_row + 1
            for i in range(data_start, df.shape[0]):
                if cell_to_str(df.iloc[i, 0]):
                    data_start = i
                    break
            
            # Build column metadata
            col_meta = []
            for col_idx in range(2, df.shape[1]):
                header = cell_to_str(df.iloc[header_row, col_idx]).upper()
                
                if not header:
                    continue
                
                # Determine policy type
                if "PAYOUT" in header and "NET" in header:
                    policy_type = "COMP"
                elif "SOD" in header or "SAOD" in header:
                    policy_type = "SAOD"
                else:
                    continue
                
                col_meta.append({
                    "col_idx": col_idx,
                    "header": cell_to_str(df.iloc[header_row, col_idx]),
                    "policy_type": policy_type,
                })
            
            if not col_meta:
                print("[TW_COMP] No data columns found")
                return records
            
            print(f"[TW_COMP] Found {len(col_meta)} columns")
            
            # Process data rows
            lob_final = override_lob if override_enabled and override_lob else "TW"
            segment_final = override_segment if override_enabled and override_segment else "TW SAOD + COMP"
            
            skip_words = {"total", "grand total", "average", "sum", ""}
            
            for row_idx in range(data_start, df.shape[0]):
                geo_location = cell_to_str(df.iloc[row_idx, 0])
                tw_type = cell_to_str(df.iloc[row_idx, 1])  # Bike or Scooter
                
                if not geo_location or geo_location.lower() in skip_words:
                    continue
                
                state = map_state(geo_location)
                
                # Process each column
                for m in col_meta:
                    payin = safe_float(df.iloc[row_idx, m["col_idx"]])
                    
                    if payin is None:
                        continue
                    
                    payout, formula, explanation = calculate_payout(payin, m["policy_type"], lob_final, segment_final)
                    
                    records.append({
                        "State": state,
                        "Geo Location": geo_location,
                        "TW Type": tw_type,  # Bike or Scooter
                        "Original Segment": m["header"],
                        "Mapped Segment": segment_final,
                        "LOB": lob_final,
                        "Policy Type": m["policy_type"],
                        "Status": "STP",
                        "Payin": f"{payin:.2f}%",
                        "Calculated Payout": f"{payout:.2f}%",
                        "Formula Used": formula,
                        "Rule Explanation": explanation,
                    })
            
            print(f"[TW_COMP] Extracted {len(records)} records")
            return records
            
        except Exception as e:
            print(f"[TW_COMP] Error: {e}")
            traceback.print_exc()
            return []


# ===============================================================================
# TW SATP PROCESSOR
# ===============================================================================

class TWSATPProcessor:
    """Process TW SATP sheets."""
    
    @staticmethod
    def process(content: bytes, sheet_name: str,
                override_enabled: bool = False,
                override_lob: str = None,
                override_segment: str = None) -> List[Dict]:
        """
        Process TW SATP pattern:
        Row 1: Title (JAN 2025 PAYOUT_TW SATP)
        Row 2: Segment (Bikes | Scooter)
        Row 3: Geo Location - New | CC bands
        Row 4+: Data rows
        """
        records = []
        
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
            
            print(f"\n[TW_SATP] Processing sheet: {sheet_name}")
            print(f"[TW_SATP] Sheet shape: {df.shape}")
            
            # Find segment row (Bikes/Scooter)
            segment_row = None
            for i in range(min(10, df.shape[0])):
                row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
                if "BIKES" in row_text or "SCOOTER" in row_text:
                    segment_row = i
                    break
            
            if segment_row is None:
                print("[TW_SATP] Segment row not found")
                return records
            
            print(f"[TW_SATP] Found segment row at index: {segment_row}")
            
            # Next row is Geo Location row
            geo_header_row = segment_row + 1
            
            # Data starts after geo header
            data_start = geo_header_row + 1
            for i in range(data_start, df.shape[0]):
                if cell_to_str(df.iloc[i, 0]):
                    data_start = i
                    break
            
            # Build column metadata
            col_meta = []
            last_segment = ""
            
            for col_idx in range(1, df.shape[1]):
                segment = cell_to_str(df.iloc[segment_row, col_idx]).upper()
                cc_band = cell_to_str(df.iloc[geo_header_row, col_idx])
                
                if not segment and not cc_band:
                    continue
                
                # Forward fill segment (for merged cells)
                if segment and ("BIKES" in segment or "SCOOTER" in segment):
                    last_segment = segment
                
                if not last_segment:
                    continue
                
                # Build description
                tw_type = "Bikes" if "BIKES" in last_segment else "Scooter"
                segment_desc = f"{tw_type}"
                if cc_band:
                    segment_desc += f" ({cc_band})"
                
                col_meta.append({
                    "col_idx": col_idx,
                    "tw_type": tw_type,
                    "cc_band": cc_band,
                    "segment_desc": segment_desc,
                })
            
            if not col_meta:
                print("[TW_SATP] No data columns found")
                return records
            
            print(f"[TW_SATP] Found {len(col_meta)} columns")
            
            # Process data rows
            lob_final = override_lob if override_enabled and override_lob else "TW"
            segment_final = override_segment if override_enabled and override_segment else "TW TP"
            
            skip_words = {"total", "grand total", "average", "sum", ""}
            
            for row_idx in range(data_start, df.shape[0]):
                geo_location = cell_to_str(df.iloc[row_idx, 0])
                
                if not geo_location or geo_location.lower() in skip_words:
                    continue
                
                state = map_state(geo_location)
                
                # Process each column
                for m in col_meta:
                    payin = safe_float(df.iloc[row_idx, m["col_idx"]])
                    
                    if payin is None:
                        continue
                    
                    payout, formula, explanation = calculate_payout(payin, "TP", lob_final, segment_final)
                    
                    records.append({
                        "State": state,
                        "Geo Location": geo_location,
                        "TW Type": m["tw_type"],  # Bikes or Scooter
                        "Original Segment": m["segment_desc"],
                        "Mapped Segment": segment_final,
                        "LOB": lob_final,
                        "Policy Type": "TP",
                        "CC Band": m["cc_band"],
                        "Status": "STP",
                        "Payin": f"{payin:.2f}%",
                        "Calculated Payout": f"{payout:.2f}%",
                        "Formula Used": formula,
                        "Rule Explanation": explanation,
                    })
            
            print(f"[TW_SATP] Extracted {len(records)} records")
            return records
            
        except Exception as e:
            print(f"[TW_SATP] Error: {e}")
            traceback.print_exc()
            return []


# ===============================================================================
# TW SATP GEO NEW/OLD PROCESSOR
# ===============================================================================

class TWSATPGeoNewOldProcessor:
    """Process TW SATP sheets with Geo segment New and Old columns."""
    
    @staticmethod
    def process(content: bytes, sheet_name: str,
                override_enabled: bool = False,
                override_lob: str = None,
                override_segment: str = None) -> List[Dict]:
        """
        Process TW SATP with Geo New/Old pattern:
        Row 2: Title (March 2025 PAYOUT - TWSATP)
        Row 4: Bikes | Scooter
        Row 5: Geo segment New | Geo segment Old | CC bands for Bikes | CC bands for Scooter
        Row 6+: Data rows
        """
        records = []
        
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
            
            print(f"\n[TW_SATP_GEO_NEW_OLD] Processing sheet: {sheet_name}")
            print(f"[TW_SATP_GEO_NEW_OLD] Sheet shape: {df.shape}")
            
            # Find segment row (Bikes/Scooter)
            segment_row = None
            for i in range(min(10, df.shape[0])):
                row_text = " ".join(cell_to_str(v) for v in df.iloc[i]).upper()
                if "BIKES" in row_text and "SCOOTER" in row_text:
                    segment_row = i
                    break
            
            if segment_row is None:
                print("[TW_SATP_GEO_NEW_OLD] Segment row not found")
                return records
            
            print(f"[TW_SATP_GEO_NEW_OLD] Found segment row at index: {segment_row}")
            
            # Next row is Geo header row (Geo segment New | Geo segment Old | CC bands)
            geo_header_row = segment_row + 1
            
            # Data starts after geo header, skip empty rows
            data_start = geo_header_row + 1
            for i in range(data_start, df.shape[0]):
                if cell_to_str(df.iloc[i, 0]) or cell_to_str(df.iloc[i, 1]):
                    data_start = i
                    break
            
            print(f"[TW_SATP_GEO_NEW_OLD] Geo header row: {geo_header_row}, Data starts: {data_start}")
            
            # Build column metadata
            col_meta = []
            last_segment = ""
            
            # First two columns are Geo segment New and Old
            for col_idx in range(2, df.shape[1]):
                segment = cell_to_str(df.iloc[segment_row, col_idx]).upper()
                cc_band = cell_to_str(df.iloc[geo_header_row, col_idx])
                
                if not segment and not cc_band:
                    continue
                
                # Forward fill segment (for merged cells)
                if segment and ("BIKES" in segment or "SCOOTER" in segment):
                    last_segment = segment
                
                if not last_segment:
                    continue
                
                # Build description
                tw_type = "Bikes" if "BIKES" in last_segment else "Scooter"
                segment_desc = f"{tw_type}"
                if cc_band:
                    segment_desc += f" ({cc_band})"
                
                col_meta.append({
                    "col_idx": col_idx,
                    "tw_type": tw_type,
                    "cc_band": cc_band,
                    "segment_desc": segment_desc,
                })
            
            if not col_meta:
                print("[TW_SATP_GEO_NEW_OLD] No data columns found")
                return records
            
            print(f"[TW_SATP_GEO_NEW_OLD] Found {len(col_meta)} columns")
            for m in col_meta[:5]:
                print(f"  - Col {m['col_idx']}: {m['segment_desc']}")
            
            # Process data rows
            lob_final = override_lob if override_enabled and override_lob else "TW"
            segment_final = override_segment if override_enabled and override_segment else "TW TP"
            
            skip_words = {"total", "grand total", "average", "sum", ""}
            
            for row_idx in range(data_start, df.shape[0]):
                geo_new = cell_to_str(df.iloc[row_idx, 0])
                geo_old = cell_to_str(df.iloc[row_idx, 1])
                
                # Accept data if either column has content
                if not geo_new and not geo_old:
                    continue
                
                if geo_new.lower() in skip_words and geo_old.lower() in skip_words:
                    continue
                
                # Use whichever is available
                if not geo_new:
                    geo_new = geo_old
                if not geo_old:
                    geo_old = geo_new
                
                # Combine Geo New and Geo Old
                combined_location = f"{geo_new} - {geo_old}" if geo_new != geo_old else geo_new
                
                # Extract state from geo_old
                state = map_state(geo_old if geo_old else geo_new)
                
                # Process each column
                for m in col_meta:
                    payin = safe_float(df.iloc[row_idx, m["col_idx"]])
                    
                    if payin is None:
                        continue
                    
                    payout, formula, explanation = calculate_payout(payin, "TP", lob_final, segment_final)
                    
                    records.append({
                        "State": state,
                        "Geo Location": combined_location,
                        "Geo New": geo_new,
                        "Geo Old": geo_old,
                        "TW Type": m["tw_type"],  # Bikes or Scooter
                        "Original Segment": m["segment_desc"],
                        "Mapped Segment": segment_final,
                        "LOB": lob_final,
                        "Policy Type": "TP",
                        "CC Band": m["cc_band"],
                        "Status": "STP",
                        "Payin": f"{payin:.2f}%",
                        "Calculated Payout": f"{payout:.2f}%",
                        "Formula Used": formula,
                        "Rule Explanation": explanation,
                    })
            
            print(f"[TW_SATP_GEO_NEW_OLD] Extracted {len(records)} records")
            return records
            
        except Exception as e:
            print(f"[TW_SATP_GEO_NEW_OLD] Error: {e}")
            traceback.print_exc()
            return []


# ===============================================================================
# PATTERN DISPATCHER
# ===============================================================================

class TWPatternDispatcher:
    """Route to correct TW processor."""
    
    PATTERN_PROCESSORS = {
        "tw_comp": TWCompProcessor,
        "tw_satp": TWSATPProcessor,
        "tw_satp_geo_new_old": TWSATPGeoNewOldProcessor,
    }
    
    @staticmethod
    def process_sheet(content: bytes, sheet_name: str,
                      override_enabled: bool = False,
                      override_lob: str = None,
                      override_segment: str = None) -> List[Dict]:
        """Detect pattern and route to processor."""
        df_raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name, header=None)
        pattern = TWPatternDetector.detect_pattern(df_raw)
        
        print(f"\n[DISPATCHER] Detected pattern: {pattern}")
        
        processor_class = TWPatternDispatcher.PATTERN_PROCESSORS.get(pattern, TWCompProcessor)
        return processor_class.process(
            content, sheet_name,
            override_enabled, override_lob, override_segment
        )


# ===============================================================================
# API ENDPOINTS
# ===============================================================================

@app.get("/")
async def root():
    return {
        "message": "Two Wheelers Payout Processor API",
        "version": "2.0.0",
        "formula": "90% for COMP/SAOD, Tiered deduction for TP",
        "supported_lobs": ["TW"],
        "supported_segments": ["TW SAOD + COMP", "TW TP"],
        "supported_patterns": [
            "tw_comp - TW COMP (Geo Locations | Type | Payout % - Net | SOD)",
            "tw_satp - TW SATP (Segment | Geo Location - New | CC bands for Bikes/Scooter)",
            "tw_satp_geo_new_old - TW SATP with Geo segment New/Old columns + CC bands"
        ],
        "special_rules": [
            "If payin ≤ 5%, payout = 0"
        ]
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload Excel file."""
    try:
        if not file.filename.endswith((".xlsx", ".xls")):
            raise HTTPException(status_code=400, detail="Only Excel files supported")
        
        # content = await file.read()
        # xls = pd.ExcelFile(io.BytesIO(content))
        # sheets = xls.sheet_names
        import openpyxl
        content = await file.read()
        
        # Load workbook and get only VISIBLE sheets
        wb = openpyxl.load_workbook(io.BytesIO(content))
        sheets = [
            sheet.title 
            for sheet in wb.worksheets 
            if sheet.sheet_state == 'visible'
        ]
        wb.close()
        
        file_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uploaded_files[file_id] = {
            "content": content,
            "filename": file.filename,
            "sheets": sheets,
        }
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "sheets": sheets,
            "message": f"Uploaded successfully. {len(sheets)} worksheet(s) found.",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/process")
async def process_sheet(
    file_id: str,
    sheet_name: str,
    override_enabled: bool = False,
    override_lob: Optional[str] = None,
    override_segment: Optional[str] = None,
):
    """Process worksheet."""
    try:
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_data = uploaded_files[file_id]
        
        if sheet_name not in file_data["sheets"]:
            raise HTTPException(status_code=400, detail=f"Sheet '{sheet_name}' not found")
        
        records = TWPatternDispatcher.process_sheet(
            file_data["content"], 
            sheet_name,
            override_enabled, 
            override_lob, 
            override_segment,
        )
        
        if not records:
            return {
                "success": False,
                "message": "No records extracted. Check sheet structure.",
                "records": [],
                "count": 0,
            }
        
        # Summary stats
        states = {}
        policies = {}
        payins = []
        payouts = []
        
        for r in records:
            state = r.get("State", "UNKNOWN")
            states[state] = states.get(state, 0) + 1
            
            policy = r.get("Policy Type", "UNKNOWN")
            policies[policy] = policies.get(policy, 0) + 1
            
            try:
                payin_val = float(r.get("Payin", "0%").replace("%", ""))
                payout_val = float(r.get("Calculated Payout", "0%").replace("%", ""))
                payins.append(payin_val)
                payouts.append(payout_val)
            except Exception:
                pass
        
        avg_payin = round(sum(payins) / len(payins), 2) if payins else 0
        avg_payout = round(sum(payouts) / len(payouts), 2) if payouts else 0
        
        return {
            "success": True,
            "message": f"Successfully processed {len(records)} records from '{sheet_name}'",
            "records": records,
            "count": len(records),
            "summary": {
                "total_records": len(records),
                "states": dict(sorted(states.items(), key=lambda x: x[1], reverse=True)[:10]),
                "policy_types": policies,
                "average_payin": avg_payin,
                "average_payout": avg_payout,
            },
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/export")
async def export_to_excel(file_id: str, sheet_name: str, records: List[Dict]):
    """Export to Excel."""
    try:
        if not records:
            raise HTTPException(status_code=400, detail="No records to export")
        
        df = pd.DataFrame(records)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"TW_Processed_{sheet_name.replace(' ', '_')}_{timestamp}.xlsx"
        out_path = os.path.join(tempfile.gettempdir(), filename)
        
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Processed Data")
            
            worksheet = writer.sheets["Processed Data"]
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
        
        return FileResponse(
            path=out_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uploaded_files": len(uploaded_files)
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("Two Wheelers Payout Processor API - v2.0.0")
    print("Patterns: TW COMP + TW SATP + TW SATP Geo New/Old")
    print("Special Rule: Payin ≤ 5% → Payout = 0")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
