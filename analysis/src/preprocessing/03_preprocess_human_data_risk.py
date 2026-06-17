#!/usr/bin/env python3
"""
Load original human data and convert it into LLM-comparable format,
then save raw and processed forms for analyses.
"""
# ----------- packages  ------------------------------------

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../../data").resolve()
SOURCE_DIR = (SCRIPT_DIR / "../../source").resolve()
RAW_DIR = SOURCE_DIR / "risk_data/orig_human_data"
OUT_DIR = DATA_DIR / "risk_data/human_data_proc"


# ------------------------ shared mappings ------------------------

CATEGORY_MAPPINGS = {
    "BARRAT scale": {
        1: "BISn", 2: "BISm", 3: "BISm", 4: "BISm", 5: "BISa", 6: "BISa", 7: "BISn", 8: "BISn",
        9: "BISa", 10: "BISn", 11: "BISa", 12: "BISn", 13: "BISn", 14: "BISn", 15: "BISn", 16: "BISm",
        17: "BISm", 18: "BISn", 19: "BISm", 20: "BISa", 21: "BISm", 22: "BISm", 23: "BISm", 24: "BISa",
        25: "BISm", 26: "BISa", 27: "BISn", 28: "BISa", 29: "BISn", 30: "BISm",
    },
    "CARE scale": {i: cat for i, cat in zip(range(1, 20), ["CAREa"] * 9 + ["CAREs"] * 6 + ["CAREw"] * 4)},
    "DOSPERT scale": {
        1: "Dsoc", 10: "Dsoc", 16: "Dsoc", 19: "Dsoc", 23: "Dsoc", 26: "Dsoc", 34: "Dsoc", 35: "Dsoc",
        2: "Drec", 6: "Drec", 15: "Drec", 17: "Drec", 21: "Drec", 31: "Drec", 37: "Drec", 38: "Drec",
        3: "Dgam", 11: "Dgam", 22: "Dgam", 33: "Dgam",
        4: "Dhea", 8: "Dhea", 27: "Dhea", 29: "Dhea", 32: "Dhea", 36: "Dhea", 39: "Dhea", 40: "Dhea",
        5: "Deth", 9: "Deth", 12: "Deth", 13: "Deth", 14: "Deth", 20: "Deth", 25: "Deth", 28: "Deth",
        7: "Dinv", 18: "Dinv", 24: "Dinv", 30: "Dinv",
    },
    "PRI scale": {
        1: "decision", 3: "decision", 5: "decision", 7: "decision", 9: "decision", 11: "decision",
        13: "decision", 15: "decision",
        2: "certainty", 4: "certainty", 6: "certainty", 8: "certainty", 10: "certainty",
        12: "certainty", 14: "certainty", 16: "certainty",
    },
    "SOEP scale": {1: "SOEP", 2: "SOEPdri", 3: "SOEPfin", 4: "SOEPrec", 5: "SOEPocc", 6: "SOEPhea", 7: "SOEPsoc"},
    "SSSV scale": {
        3: "SStas", 11: "SStas", 16: "SStas", 17: "SStas", 20: "SStas", 21: "SStas", 23: "SStas", 28: "SStas",
        38: "SStas", 40: "SStas",
        4: "SSexp", 6: "SSexp", 9: "SSexp", 10: "SSexp", 14: "SSexp", 18: "SSexp", 19: "SSexp", 22: "SSexp",
        26: "SSexp", 37: "SSexp",
        1: "SSdis", 12: "SSdis", 13: "SSdis", 25: "SSdis", 29: "SSdis", 30: "SSdis", 32: "SSdis", 33: "SSdis",
        35: "SSdis", 36: "SSdis",
        2: "SSbor", 5: "SSbor", 7: "SSbor", 8: "SSbor", 15: "SSbor", 24: "SSbor", 27: "SSbor", 31: "SSbor",
        34: "SSbor", 39: "SSbor",
    },
}

PG_ITEM_NUMBERS = {
    "Gamblingpart": 0,
    "ESS_GABS_innerh_1": 1, "ESS_GABS_innerh_2": 2, "ESS_GABS_innerh_3": 3, "ESS_GABS_innerh_4": 4,
    "ESS_GABS_innerh_5": 5, "ESS_GABS_innerh_6": 6, "ESS_GABS_innerh_7": 7, "ESS_GABS_innerh_8": 8, "ESS_GABS_innerh_9": 9,
    "ESS_GABS_ausserh_1": 10, "ESS_GABS_ausserh_2": 11, "ESS_GABS_ausserh_3": 12, "ESS_GABS_ausserh_4": 13,
    "ESS_GABS_ausserh_5": 14, "ESS_GABS_ausserh_6": 15, "ESS_GABS_ausserh_7": 16, "ESS_GABS_ausserh_8": 17,
    "ESS_GABS_ausserh_9": 18, "ESS_GABS_ausserh_10": 19, "ESS_GABS_ausserh_11": 20, "ESS_GABS_ausserh_12": 21,
    "ESS_GABS_ausserh_13": 22, "ESS_GABS_ausserh_14": 23, "ESS_GABS_ausserh_15": 24, "ESS_GABS_ausserh_16": 25,
    "ESS_GABS_ausserh_17": 26, "ESS_GABS_ausserh_18": 27, "ESS_GABS_ausserh_19": 28, "ESS_GABS_ausserh_20": 29,
    "ESS_GABS_ausserh_21": 23, "ESS_GABS_ausserh_22": 31,
}


def _extract_item_number_default(name: str) -> int | None:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else None


def _extract_item_number_dm_proc(name: str, exp: str) -> int | None:
    if exp == "DM scale":
        m = re.search(r"Dy(\d+)\.1", name)
        return int(m.group(1)) if m else None
    return _extract_item_number_default(name)


# ------------------------ mode configs ------------------------

RAW_CONFIG = {
    "name": "raw",
    "survey_file": RAW_DIR / "quest_raw.csv",
    "drop_columns": [
        "duration", "location", "v_943", "soziooekonomM", "soziooekonomV", "soziooekonomSelbst",
        "v_944", "BerlinNumeracy1", "BerlinNumeracy2a", "BerlinNumeracy2b", "BerlinNumeracy3",
        "RLRQ_Risiko_Bank_15", "RLRQ_Risiko_Altersvorsorge_16",
        "DOSPERTrisk_1", "DOSPERTrisk_2", "DOSPERTrisk_3", "DOSPERTrisk_4", "DOSPERTrisk_5", "DOSPERTrisk_6",
        "DOSPERTrisk_7", "DOSPERTrisk_8", "DOSPERTrisk_9", "DOSPERTrisk_10", "DOSPERTrisk_11", "DOSPERTrisk_12",
        "DOSPERTrisk_13", "DOSPERTrisk_14", "DOSPERTrisk_15", "DOSPERTrisk_16", "DOSPERTrisk_17", "DOSPERTrisk_18",
        "DOSPERTrisk_19", "DOSPERTrisk_20", "DOSPERTrisk_21", "DOSPERTrisk_22", "DOSPERTrisk_23", "DOSPERTrisk_24",
        "DOSPERTrisk_25", "DOSPERTrisk_26", "DOSPERTrisk_27", "DOSPERTrisk_28", "DOSPERTrisk_29", "DOSPERTrisk_30",
        "DOSPERTrisk_31", "DOSPERTrisk_32", "DOSPERTrisk_33", "DOSPERTrisk_34", "DOSPERTrisk_35", "DOSPERTrisk_36",
        "DOSPERTrisk_37", "DOSPERTrisk_38", "DOSPERTrisk_39", "DOSPERTrisk_40",
        "DOSPERTnutz_1", "DOSPERTnutz_2", "DOSPERTnutz_3", "DOSPERTnutz_4", "DOSPERTnutz_5", "DOSPERTnutz_6",
        "DOSPERTnutz_7", "DOSPERTnutz_8", "DOSPERTnutz_9", "DOSPERTnutz_10", "DOSPERTnutz_11", "DOSPERTnutz_12",
        "DOSPERTnutz_13", "DOSPERTnutz_14", "DOSPERTnutz_15", "DOSPERTnutz_16", "DOSPERTnutz_17", "DOSPERTnutz_18",
        "DOSPERTnutz_19", "DOSPERTnutz_20", "DOSPERTnutz_21", "DOSPERTnutz_22", "DOSPERTnutz_23", "DOSPERTnutz_24",
        "DOSPERTnutz_25", "DOSPERTnutz_26", "DOSPERTnutz_27", "DOSPERTnutz_28", "DOSPERTnutz_29", "DOSPERTnutz_30",
        "DOSPERTnutz_31", "DOSPERTnutz_32", "DOSPERTnutz_33", "DOSPERTnutz_34", "DOSPERTnutz_35", "DOSPERTnutz_36",
        "DOSPERTnutz_37", "DOSPERTnutz_38", "DOSPERTnutz_39", "DOSPERTnutz_40",
        "Familienstand", "Geschwister", "Geschwister_aelter", "Geschwister_juenger", "Schulabschluss",
        "Ausbildung_1", "Ausbildung_2", "Ausbildung_3", "Ausbildung_4", "Haushalt", "Sozio_Status",
        "Familienstand_Berlin", "Geschwister_Berlin", "Geschwister_aelter_Berlin", "Geschwister_juenger_Berlin",
        "Schulabschluss_Berlin", "Ausbildung_1_Berlin", "Ausbildung_2_Berlin", "Ausbildung_3_Berlin",
        "Ausbildung_4_Berlin", "Haushalt_Berlin", "Sozio_Status_Berlin",
        "Gesundheit01", "Gesundheit02", "Gesundheit03", "Gesundheit04", "Gesundheit05", "Gesundheit06",
        "Gesundheit07", "Gesundheit08", "Gesundheit09", "Gesundheit10", "Gesundheit11", "Gesundheit12",
        "Gesundheit13", "Gesundheit14", "Gesundheit15", "Gesundheit16", "Gesundheit17", "Gesundheit18",
        "Gesundheit19", "Gesundheit20", "Gesundheit21", "Gesundheit22", "Gesundheit23", "Gesundheit24",
        "Gesundheit25", "GesundheitGewicht", "GesundheitGroesse", "GesundheitMomVerfassung", "GesundheitVglVerfassung",
        "LifeEvents01", "LifeEvents01belastung", "LifeEvents02", "LifeEvents02belastung", "LifeEvents03",
        "LifeEvents03belastung", "LifeEvents04", "LifeEvents04belastung", "LifeEvents05", "LifeEvents05belastung",
        "LifeEvents06", "LifeEvents06belastung", "LifeEvents07", "LifeEvents07belastung", "LifeEvents08",
        "LifeEvents08belastung", "LifeEvents09", "LifeEvents09belastung", "LifeEvents10", "LifeEvents10belastung",
        "LifeEvents11", "LifeEvents11belastung", "LifeEvents12", "LifeEvents12belastung", "LifeEvents13",
        "LifeEvents13belastung", "LifeEvents14", "LifeEvents14belastung", "LifeEvents15", "LifeEvents15belastung",
        "LifeEvents16", "LifeEvents16belastung", "LifeEvents17", "LifeEvents17belastung", "LifeEvents18",
        "LifeEvents18belastung", "LifeEvents19", "LifeEvents19belastung", "LifeEvents20", "LifeEvents20belastung",
        "LifeEvents21", "LifeEvents21belastung", "LifeEvents22", "LifeEvents22belastung", "LifeEvents23",
        "LifeEvents23belastung", "LifeEvents24", "LifeEvents24belastung", "LifeEvents25", "LifeEvents25belastung",
        "LifeEvents26", "LifeEvents26belastung", "LifeEvents27", "LifeEvents27belastung", "LifeEvents28",
        "LifeEvents28belastung", "LifeEvents29", "LifeEvents29belastung", "LifeEvents30", "LifeEvents30belastung",
        "LifeEvents31", "LifeEvents31belastung", "LifeEvents32belastung", "LifeEvents33belastung",
        "LifeEvents34belastung", "LifeEvents35belastung",
        "Chrono_2", "Chrono_3", "Chrono_5", "Chrono_7", "Chrono_8", "Chrono_9", "Chrono_10", "Chrono_11",
        "Chrono_12", "Chrono14", "Chrono_15", "Chrono16", "Chrono_17", "Chrono_18", "Chrono_20",
        "Chrono_21", "v_937", "v_938", "Interesse_Folgestudien", "TNaehnlicheExp",
    ],
    "experiment_mapping": {
        "AUDIT scale": [f"AUDIT_{i}" for i in range(1, 11)],
        "FTND scale": ["FTND_Eingangsfrage"] + [f"FTND_{i}" for i in range(1, 7)],
        "GABS scale": ["Gamblingpart"] + [f"ESS_GABS_ausserh_{i}" for i in range(23, 38)],
        "PG scale": ["Gamblingpart"] + [f"ESS_GABS_innerh_{i}" for i in range(1, 10)] + [f"ESS_GABS_ausserh_{i}" for i in range(1, 22)],
        "CARE scale": [f"RLRQ2_{i}" for i in range(1, 20)],
        "DM scale": [f"DOSPERTmonat_{i}" for i in range(1, 19)],
        "BARRAT scale": [f"BARRATimp{i:02d}" for i in range(1, 31)],
        "SSSV scale": [f"SSSV{i:02d}" for i in range(1, 41)],
        "PRI scale": [f"PRI_{i}" for i in range(1, 17)],
        "DOSPERT scale": [f"DOSPERT_{i}" for i in range(1, 41)],
        "SOEP scale": ["RLRQ_1", "RLRQ_3", "RLRQ_4", "RLRQ_5", "RLRQ_6", "RLRQ_7", "RLRQ_8"],
        "DAST scale": [f"DAST_{i}" for i in range(1, 21)],
    },
    "item_number_maps": {
        "PG scale": PG_ITEM_NUMBERS,
        "SOEP scale": {"RLRQ_1": 1, "RLRQ_3": 2, "RLRQ_4": 3, "RLRQ_5": 4, "RLRQ_6": 5, "RLRQ_7": 6, "RLRQ_8": 7},
        "FTND scale": {"SmokeSplit": 1, "FTND_1": 2, "FTND_2": 3, "FTND_3": 4, "FTND_4": 5, "FTND_5": 6, "FTND_6": 7},
    },
    "extract_item_number": lambda name, _exp: _extract_item_number_default(name),
    "output_file": OUT_DIR / "raw_items_per_person.csv",
}

PROC_CONFIG = {
    "name": "proc",
    "survey_file": RAW_DIR / "quest_proc.csv",
    "drop_columns": [
        "duration", "location", "ses_moth", "v_943", "ses_fath", "v_944", "ses", "BMI", "famstat", "sibl",
        "sibl_y", "sibl_o", "birthrank", "edu", "household", "birthplace", "income", "SSSV", "weight", "height",
        "BMI_raw", "BerlinNumeracy1", "BerlinNumeracy2a", "BerlinNumeracy2b", "BerlinNumeracy3", "SOEP_COMP",
        "SOEPdri_COMP", "SOEPfin_COMP", "SOEPrec_COMP", "SOEPocc_COMP", "SOEPhea_COMP", "SOEPsoc_COMP",
        "RLRQ_Risiko_Bank_15", "RLRQ_Risiko_Altersvorsorge_16",
        "DOSPERTrisk_1", "DOSPERTrisk_2", "DOSPERTrisk_3", "DOSPERTrisk_4", "DOSPERTrisk_5", "DOSPERTrisk_6",
        "DOSPERTrisk_7", "DOSPERTrisk_8", "DOSPERTrisk_9", "DOSPERTrisk_10", "DOSPERTrisk_11", "DOSPERTrisk_12",
        "DOSPERTrisk_13", "DOSPERTrisk_14", "DOSPERTrisk_15", "DOSPERTrisk_16", "DOSPERTrisk_17", "DOSPERTrisk_18",
        "DOSPERTrisk_19", "DOSPERTrisk_20", "DOSPERTrisk_21", "DOSPERTrisk_22", "DOSPERTrisk_23", "DOSPERTrisk_24",
        "DOSPERTrisk_25", "DOSPERTrisk_26", "DOSPERTrisk_27", "DOSPERTrisk_28", "DOSPERTrisk_29", "DOSPERTrisk_30",
        "DOSPERTrisk_31", "DOSPERTrisk_32", "DOSPERTrisk_33", "DOSPERTrisk_34", "DOSPERTrisk_35", "DOSPERTrisk_36",
        "DOSPERTrisk_37", "DOSPERTrisk_38", "DOSPERTrisk_39", "DOSPERTrisk_40",
        "DOSPERTnutz_1", "DOSPERTnutz_2", "DOSPERTnutz_3", "DOSPERTnutz_4", "DOSPERTnutz_5", "DOSPERTnutz_6",
        "DOSPERTnutz_7", "DOSPERTnutz_8", "DOSPERTnutz_9", "DOSPERTnutz_10", "DOSPERTnutz_11", "DOSPERTnutz_12",
        "DOSPERTnutz_13", "DOSPERTnutz_14", "DOSPERTnutz_15", "DOSPERTnutz_16", "DOSPERTnutz_17", "DOSPERTnutz_18",
        "DOSPERTnutz_19", "DOSPERTnutz_20", "DOSPERTnutz_21", "DOSPERTnutz_22", "DOSPERTnutz_23", "DOSPERTnutz_24",
        "DOSPERTnutz_25", "DOSPERTnutz_26", "DOSPERTnutz_27", "DOSPERTnutz_28", "DOSPERTnutz_29", "DOSPERTnutz_30",
        "DOSPERTnutz_31", "DOSPERTnutz_32", "DOSPERTnutz_33", "DOSPERTnutz_34", "DOSPERTnutz_35", "DOSPERTnutz_36",
        "DOSPERTnutz_37", "DOSPERTnutz_38", "DOSPERTnutz_39", "DOSPERTnutz_40",
        "Dy1", "Dy2", "Dy3", "Dy4", "Dy5", "Dy6", "Dy7", "Dy8", "Dy9", "Dy10", "Dy11", "Dy12",
        "Dy13", "Dy14", "Dy15", "Dy16", "Dy17", "Dy18", "Dy19", "Dy20", "Dy21", "Dy22", "Dy23",
        "Dy24", "Dy25", "Dy26", "Dy27",
        "Familienstand", "Geschwister", "Geschwister_aelter", "Geschwister_juenger", "Schulabschluss", "Ausbildung_1",
        "Ausbildung_2", "Ausbildung_3", "Ausbildung_4", "Haushalt", "Sozio_Status", "Familienstand_Berlin",
        "Geschwister_Berlin", "Geschwister_aelter_Berlin", "Geschwister_juenger_Berlin", "Schulabschluss_Berlin",
        "Ausbildung_1_Berlin", "Ausbildung_2_Berlin", "Ausbildung_3_Berlin", "Ausbildung_4_Berlin",
        "Haushalt_Berlin", "Sozio_Status_Berlin",
        "Gesundheit01", "Gesundheit02", "Gesundheit03", "Gesundheit04", "Gesundheit05", "Gesundheit06",
        "Gesundheit07", "Gesundheit08", "Gesundheit09", "Gesundheit10", "Gesundheit11", "Gesundheit12",
        "Gesundheit13", "Gesundheit14", "Gesundheit15", "Gesundheit16", "Gesundheit17", "Gesundheit18",
        "Gesundheit19", "Gesundheit20", "Gesundheit21", "Gesundheit22", "Gesundheit23", "Gesundheit24",
        "Gesundheit25", "GesundheitGewicht", "GesundheitGroesse", "GesundheitMomVerfassung", "GesundheitVglVerfassung",
        "LifeEvents01", "LifeEvents01belastung", "LifeEvents02", "LifeEvents02belastung", "LifeEvents03",
        "LifeEvents03belastung", "LifeEvents04", "LifeEvents04belastung", "LifeEvents05", "LifeEvents05belastung",
        "LifeEvents06", "LifeEvents06belastung", "LifeEvents07", "LifeEvents07belastung", "LifeEvents08",
        "LifeEvents08belastung", "LifeEvents09", "LifeEvents09belastung", "LifeEvents10", "LifeEvents10belastung",
        "LifeEvents11", "LifeEvents11belastung", "LifeEvents12", "LifeEvents12belastung", "LifeEvents13",
        "LifeEvents13belastung", "LifeEvents14", "LifeEvents14belastung", "LifeEvents15", "LifeEvents15belastung",
        "LifeEvents16", "LifeEvents16belastung", "LifeEvents17", "LifeEvents17belastung", "LifeEvents18",
        "LifeEvents18belastung", "LifeEvents19", "LifeEvents19belastung", "LifeEvents20", "LifeEvents20belastung",
        "LifeEvents21", "LifeEvents21belastung", "LifeEvents22", "LifeEvents22belastung", "LifeEvents23",
        "LifeEvents23belastung", "LifeEvents24", "LifeEvents24belastung", "LifeEvents25", "LifeEvents25belastung",
        "LifeEvents26", "LifeEvents26belastung", "LifeEvents27", "LifeEvents27belastung", "LifeEvents28",
        "LifeEvents28belastung", "LifeEvents29", "LifeEvents29belastung", "LifeEvents30", "LifeEvents30belastung",
        "LifeEvents31", "LifeEvents31belastung", "LifeEvents32belastung", "LifeEvents33belastung",
        "LifeEvents34belastung", "LifeEvents35belastung",
        "Chrono_2", "Chrono_3", "Chrono_5", "Chrono_7", "Chrono_8", "Chrono_9", "Chrono_10", "Chrono_11",
        "Chrono_12", "Chrono14", "Chrono_15", "Chrono16", "Chrono_17", "Chrono_18", "Chrono_20",
        "Chrono_21", "v_937", "v_938", "Interesse_Folgestudien", "TNaehnlicheExp",
        # outcome variables
        "AUDIT_raw", "AUDIT", "FTND_raw", "FTND", "GABS_raw", "GABS", "PG_raw", "PG", "DAST_raw", "DAST", "NUM",
        "CAREaggr_raw", "CAREsex_raw", "CAREwork_raw", "CAREaggr", "CAREsex", "CAREwork", "Deth", "Dinv", "Dgam",
        "Dhea", "Drec", "Dsoc", "Deth_r", "Dinv_r", "Dgam_r", "Dhea_r", "Drec_r", "Dsoc_r", "Deth_b", "Dinv_b",
        "Dgam_b", "Dhea_b", "Drec_b", "Dsoc_b", "Dy", "Dm", "PRI_1_weighted", "PRI_3_weighted", "PRI_5_weighted",
        "PRI_7_weighted", "PRI_9_weighted", "PRI_11_weighted", "PRI_13_weighted", "PRI_15_weighted", "PRI", "PRIw",
        "BIS1att", "BIS1mot", "BIS1ctr", "BIS1com", "BIS1per", "BIS1ins", "BIS2att", "BIS2mot", "BIS2npl",
        "BIS", "SStas", "SSexp", "SSdis", "SSbor",
    ],
    "experiment_mapping": {
        "AUDIT scale": ["AlcSplit"] + [f"AUDIT_{i}" for i in range(1, 11)],
        "FTND scale": ["SmokeSplit"] + [f"FTND_{i}" for i in range(1, 7)],
        "GABS scale": ["Gamblingpart"] + [f"ESS_GABS_ausserh_{i}" for i in range(23, 38)],
        "PG scale": ["Gamblingpart"] + [f"ESS_GABS_innerh_{i}" for i in range(1, 10)] + [f"ESS_GABS_ausserh_{i}" for i in range(1, 22)],
        "CARE scale": [f"RLRQ2_{i}" for i in range(1, 20)],
        "DM scale": [f"Dy{i}.1" for i in range(1, 19)],
        "BARRAT scale": [f"BARRATimp{i:02d}" for i in range(1, 31)],
        "SSSV scale": [f"SSSV{i:02d}" for i in range(1, 41)],
        "PRI scale": [f"PRI_{i}" for i in range(1, 17)],
        "DOSPERT scale": [f"DOSPERT_{i}" for i in range(1, 41)],
        "SOEP scale": ["SOEP", "SOEPdri", "SOEPfin", "SOEPrec", "SOEPocc", "SOEPhea", "SOEPsoc"],
        "DAST scale": [f"DAST_{i}" for i in range(1, 21)],
    },
    "item_number_maps": {
        "PG scale": PG_ITEM_NUMBERS,
        "SOEP scale": {"SOEP": 1, "SOEPdri": 2, "SOEPfin": 3, "SOEPrec": 4, "SOEPocc": 5, "SOEPhea": 6, "SOEPsoc": 7},
        "AUDIT scale": {"AlcSplit": 1, "AUDIT_1": 2, "AUDIT_2": 3, "AUDIT_3": 4, "AUDIT_4": 5, "AUDIT_5": 6, "AUDIT_6": 7,
                        "AUDIT_7": 8, "AUDIT_8": 9, "AUDIT_9": 10, "AUDIT_10": 11},
        "FTND scale": {"SmokeSplit": 1, "FTND_1": 2, "FTND_2": 3, "FTND_3": 4, "FTND_4": 5, "FTND_5": 6, "FTND_6": 7},
    },
    "extract_item_number": lambda name, exp: _extract_item_number_dm_proc(name, exp),
    "output_file": OUT_DIR / "items_per_person.csv",
}


# ------------------------ pipeline functions ------------------------


def load_task_data() -> dict[str, pd.DataFrame]:
    return {
        "bart": pd.read_csv(RAW_DIR / "bart_pumps.csv"),
        "cct": pd.read_csv(RAW_DIR / "cct.csv"),
        "dfd": pd.read_csv(RAW_DIR / "dfd_perprob.csv"),
        "dfe": pd.read_csv(RAW_DIR / "dfe_perprob.csv"),
        "mpl": pd.read_csv(RAW_DIR / "mpl.csv"),
        "lot": pd.read_csv(RAW_DIR / "lotteries.csv"),
    }


def preprocess_survey_data(survey_data: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    return survey_data.drop(columns=drop_columns, errors="ignore")


def build_long_survey(survey_data: pd.DataFrame, config: dict) -> pd.DataFrame:
    long_list: list[pd.DataFrame] = []
    experiment_mapping: dict[str, list[str]] = config["experiment_mapping"]
    item_number_maps: dict[str, dict] = config["item_number_maps"]
    extract_item_number = config["extract_item_number"]

    for exp, cols in experiment_mapping.items():
        cols_clean = [
            c.replace("_Berlin", "") # Remove _Berlin suffix if present
            for c in cols
            if c in survey_data.columns or c.replace("_Berlin", "") in survey_data.columns
        ]
        df_subset = survey_data[["partid"] + cols_clean].copy()
        df_long = df_subset.melt(id_vars="partid", var_name="item", value_name="score")
        df_long["experiment"] = exp

        if exp in item_number_maps:
            df_long["item"] = df_long["item"].map(item_number_maps[exp])
        else:
            df_long["item"] = df_long["item"].apply(lambda x: extract_item_number(x, exp))

        if exp in CATEGORY_MAPPINGS:
            df_long["category"] = df_long["item"].map(CATEGORY_MAPPINGS[exp])
        else:
            df_long["category"] = None

        long_list.append(df_long)

    long_survey = pd.concat(long_list, ignore_index=True)
    long_survey = long_survey[["experiment", "partid", "item", "score", "category"]]
    return long_survey


def append_task_data(long_survey: pd.DataFrame, task_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bart_data = task_data["bart"][["partid", "trial", "pumps"]].rename(columns={"trial": "item", "pumps": "score"})
    bart_data["experiment"] = "BART task"
    all_data = pd.concat([long_survey, bart_data], ignore_index=True)

    cct_data = task_data["cct"][["partid", "r_trialnum", "r_cardschosen"]].rename(columns={"r_trialnum": "item", "r_cardschosen": "score"})
    cct_data["experiment"] = "CCT task"
    all_data = pd.concat([all_data, cct_data], ignore_index=True)

    dfd_data = task_data["dfd"][["partid", "gamble_lab", "R"]].rename(columns={"gamble_lab": "item", "R": "score"})
    dfd_data["experiment"] = "DFD task"
    all_data = pd.concat([all_data, dfd_data], ignore_index=True)

    dfe_data = task_data["dfe"][["partid", "gamble_lab", "Rexp"]]
    dfe_data = dfe_data.rename(columns={"gamble_lab": "item", "Rexp": "score"})
    dfe_data["experiment"] = "DFE task"
    all_data = pd.concat([all_data, dfe_data], ignore_index=True)

    lot_data = task_data["lot"][["partid", "Dec_ID", "R"]].rename(columns={"Dec_ID": "item", "R": "score"})
    lot_data["experiment"] = "LOT task"
    all_data = pd.concat([all_data, lot_data], ignore_index=True)

    mpl_data = task_data["mpl"].copy()
    mpl_data["item"] = list(zip(mpl_data["dp"], mpl_data["decision"]))
    mpl_data = mpl_data[["partid", "item", "R"]].rename(columns={"R": "score"})
    mpl_data["experiment"] = "MPL task"
    all_data = pd.concat([all_data, mpl_data], ignore_index=True)

    return all_data


def run_pipeline(config: dict) -> Path:
    survey_data = pd.read_csv(config["survey_file"])
    survey_data = preprocess_survey_data(survey_data, config["drop_columns"])
    long_survey = build_long_survey(survey_data, config)
    all_data = append_task_data(long_survey, load_task_data())

    output_file: Path = config["output_file"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_data.to_csv(output_file, index=False)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess human risk data (raw and processed).")
    parser.add_argument("--mode", choices=["raw", "proc", "both"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode

    outputs: list[Path] = []
    if mode in {"raw", "both"}:
        outputs.append(run_pipeline(RAW_CONFIG))
    if mode in {"proc", "both"}:
        outputs.append(run_pipeline(PROC_CONFIG))

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
