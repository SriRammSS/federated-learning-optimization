from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize as _winsorize
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .io import write_csv,write_json
from .data import ClientData


@dataclass(frozen=True)
class MimicConfig:
    data_dir:Path=Path("data")
    out:Path=Path("outputs/full_mimic_iv")
    db_path:Path=Path("data/mimic_cache/mimic_iv.duckdb")
    hours:int=24
    top_chart:int=50
    top_labs:int=50
    top_inputs:int=30
    top_outputs:int=20
    top_procedures:int=25
    top_rx:int=30
    seed:int=7
    threads:int=12
    memory_limit:str="36GB"


@dataclass
class MimicBundle:
    clients:list[ClientData]
    feature_names:list[str]
    class_names:list[str]
    client_names:dict[int,str]
    root:Path
    arrays_path:Path
    class_weights:tuple[float,float]


def find_mimic_root(data_dir:Path)->Path:
    candidates=sorted(data_dir.glob("**/mimic-iv-2.1"))
    if not candidates:
        raise FileNotFoundError(f"mimic-iv-2.1 not found under {data_dir}")
    return candidates[0]


def load_mimic_iv_arrays(out_dir:Path=Path("outputs/full_mimic_iv"),seed:int=7)->MimicBundle:
    arrays_path=out_dir/"preprocessing"/"model_arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"missing MIMIC arrays: {arrays_path}")
    arr=np.load(arrays_path,allow_pickle=True)
    x=arr["x"].astype("float32")
    y=arr["y"].astype("int64")
    client_ids=arr["client_id"].astype("int64")
    splits=arr["split"].astype(str)
    feature_names=[str(v) for v in arr["feature_names"].tolist()]
    client_names=_read_client_names(out_dir)
    clients=[]
    for cid in sorted(set(client_ids.tolist())):
        train=(client_ids==cid)&(splits=="train")
        test=(client_ids==cid)&(splits=="test")
        clients.append(ClientData(x[train],y[train],x[test],y[test],int(cid)))
    counts=np.bincount(y[splits=="train"],minlength=2).astype(float)
    weights=(counts.sum()/(2*np.maximum(counts,1))).astype(float)
    return MimicBundle(clients,feature_names,["survived","expired"],client_names,out_dir,arrays_path,(float(weights[0]),float(weights[1])))


def _read_client_names(out_dir:Path)->dict[int,str]:
    path=out_dir/"preprocessing"/"client_map.csv"
    if not path.exists():
        return {}
    df=pd.read_csv(path)
    return {int(r.client_id):str(r.client_name) for r in df.itertuples()}


def build_mimic_preprocessing(cfg:MimicConfig)->dict:
    cfg.out.mkdir(parents=True,exist_ok=True)
    for name in ["preprocessing","eda","runtime","artifacts","plots/eda","plots/preprocessing"]:
        (cfg.out/name).mkdir(parents=True,exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True,exist_ok=True)
    root=find_mimic_root(cfg.data_dir)
    con=duckdb.connect(str(cfg.db_path))
    con.execute(f"PRAGMA threads={cfg.threads}")
    con.execute(f"PRAGMA memory_limit='{cfg.memory_limit}'")
    runtime=[]
    try:
        _step("cohort",runtime,lambda:_build_cohort(con,root,cfg.hours))
        _step("chart_events",runtime,lambda:_build_numeric_events(con,root,"chartevents","icu/chartevents.csv","stay_id","charttime","valuenum","d_items","icu/d_items.csv",cfg.top_chart,cfg.hours))
        _step("lab_events",runtime,lambda:_build_numeric_events(con,root,"labevents","hosp/labevents.csv","hadm_id","charttime","valuenum","d_labitems","hosp/d_labitems.csv",cfg.top_labs,cfg.hours))
        _step("icu_inputs",runtime,lambda:_build_inputs(con,root,cfg.top_inputs,cfg.hours))
        _step("icu_outputs",runtime,lambda:_build_outputs(con,root,cfg.top_outputs,cfg.hours))
        _step("icu_procedures",runtime,lambda:_build_procedure_events(con,root,cfg.top_procedures,cfg.hours))
        _step("prescriptions",runtime,lambda:_build_prescriptions(con,root,cfg.top_rx,cfg.hours))
        _step("admin_counts",runtime,lambda:_build_admin_counts(con,root,cfg.hours))
        _step("feature_export",runtime,lambda:_export_features(con,cfg,root))
        _step("eda",runtime,lambda:_write_eda(con,cfg))
    finally:
        con.close()
    write_csv(cfg.out/"runtime"/"preprocessing_runtime.csv",runtime)
    meta={
        "mimic_root":str(root),
        "db_path":str(cfg.db_path),
        "hours":cfg.hours,
        "top_chart":cfg.top_chart,
        "top_labs":cfg.top_labs,
        "top_inputs":cfg.top_inputs,
        "top_outputs":cfg.top_outputs,
        "top_procedures":cfg.top_procedures,
        "top_rx":cfg.top_rx,
        "seed":cfg.seed,
        "threads":cfg.threads,
        "memory_limit":cfg.memory_limit,
    }
    write_json(cfg.out/"preprocessing"/"preprocessing_metadata.json",meta)
    return meta


def _step(name:str,runtime:list[dict],fn):
    start=time.perf_counter()
    fn()
    runtime.append({"stage":name,"seconds":time.perf_counter()-start})


def _csv(root:Path,rel:str)->str:
    return str(root/rel).replace("'","''")


def _read(root:Path,rel:str)->str:
    return f"read_csv('{_csv(root,rel)}',header=true,all_varchar=true,ignore_errors=true)"


def _build_cohort(con,root:Path,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE cohort AS
        WITH icu AS (
            SELECT
                TRY_CAST(subject_id AS BIGINT) subject_id,
                TRY_CAST(hadm_id AS BIGINT) hadm_id,
                TRY_CAST(stay_id AS BIGINT) stay_id,
                first_careunit,
                last_careunit,
                TRY_CAST(intime AS TIMESTAMP) intime,
                TRY_CAST(outtime AS TIMESTAMP) outtime
            FROM {_read(root,'icu/icustays.csv')}
        ),
        adm AS (
            SELECT
                TRY_CAST(subject_id AS BIGINT) subject_id,
                TRY_CAST(hadm_id AS BIGINT) hadm_id,
                TRY_CAST(admittime AS TIMESTAMP) admittime,
                TRY_CAST(edregtime AS TIMESTAMP) edregtime,
                TRY_CAST(edouttime AS TIMESTAMP) edouttime,
                admission_type,
                admission_location,
                insurance,
                language,
                marital_status,
                race,
                TRY_CAST(hospital_expire_flag AS INTEGER) mortality_label
            FROM {_read(root,'hosp/admissions.csv')}
        ),
        pat AS (
            SELECT
                TRY_CAST(subject_id AS BIGINT) subject_id,
                gender,
                TRY_CAST(anchor_age AS DOUBLE) anchor_age
            FROM {_read(root,'hosp/patients.csv')}
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY i.stay_id)-1 row_id,
            i.subject_id,
            i.hadm_id,
            i.stay_id,
            i.first_careunit client_name,
            DENSE_RANK() OVER (ORDER BY i.first_careunit)-1 client_id,
            i.last_careunit,
            i.intime,
            i.outtime,
            a.admittime,
            a.edregtime,
            a.edouttime,
            a.admission_type,
            a.admission_location,
            a.insurance,
            a.language,
            a.marital_status,
            a.race,
            p.gender,
            p.anchor_age,
            a.mortality_label,
            DATE_DIFF('hour',a.admittime,i.intime) hours_hosp_before_icu,
            DATE_DIFF('hour',a.edregtime,a.edouttime) ed_hours,
            EXTRACT(hour FROM i.intime) icu_admit_hour,
            EXTRACT(dow FROM i.intime) icu_admit_dow,
            CASE WHEN p.gender='M' THEN 1 ELSE 0 END gender_m,
            CASE WHEN p.gender='F' THEN 1 ELSE 0 END gender_f,
            CASE WHEN i.outtime>=i.intime+INTERVAL {hours} HOUR THEN 1 ELSE 0 END observed_full_window
        FROM icu i
        JOIN adm a ON i.subject_id=a.subject_id AND i.hadm_id=a.hadm_id
        JOIN pat p ON i.subject_id=p.subject_id
        WHERE i.subject_id IS NOT NULL
            AND i.hadm_id IS NOT NULL
            AND i.stay_id IS NOT NULL
            AND i.intime IS NOT NULL
            AND i.first_careunit IS NOT NULL
            AND a.mortality_label IN (0,1)
    """)
    con.execute("CREATE OR REPLACE TABLE client_map AS SELECT DISTINCT client_id,client_name FROM cohort ORDER BY client_id")


def _build_numeric_events(con,root:Path,name:str,rel:str,join_key:str,time_col:str,value_col:str,dict_name:str,dict_rel:str,top_n:int,hours:int)->None:
    con.execute(f"CREATE OR REPLACE TABLE {dict_name} AS SELECT * FROM {_read(root,dict_rel)}")
    join_expr=f"TRY_CAST(e.{join_key} AS BIGINT)=c.{join_key}"
    con.execute(f"""
        CREATE OR REPLACE TABLE {name}_24h AS
        SELECT
            c.stay_id,
            TRY_CAST(e.itemid AS BIGINT) itemid,
            TRY_CAST(e.{value_col} AS DOUBLE) valuenum,
            TRY_CAST(e.{time_col} AS TIMESTAMP) event_time
        FROM {_read(root,rel)} e
        JOIN cohort c ON {join_expr}
        WHERE TRY_CAST(e.{value_col} AS DOUBLE) IS NOT NULL
            AND TRY_CAST(e.{time_col} AS TIMESTAMP)>=c.intime
            AND TRY_CAST(e.{time_col} AS TIMESTAMP)<c.intime+INTERVAL {hours} HOUR
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {name}_item_counts AS
        SELECT e.itemid,ANY_VALUE(d.label) item_label,COUNT(*) event_rows,COUNT(DISTINCT e.stay_id) stays
        FROM {name}_24h e
        LEFT JOIN {dict_name} d ON TRY_CAST(d.itemid AS BIGINT)=e.itemid
        GROUP BY e.itemid
        ORDER BY event_rows DESC
        LIMIT {top_n}
    """)
    _wide_numeric(con,f"{name}_features",f"{name}_24h",f"{name}_item_counts",name)


def _wide_numeric(con,out_table:str,event_table:str,top_table:str,prefix:str)->None:
    itemids=[int(r[0]) for r in con.execute(f"SELECT itemid FROM {top_table} ORDER BY event_rows DESC").fetchall()]
    exprs=["COUNT(*) total_events","COUNT(DISTINCT itemid) distinct_items"]
    for itemid in itemids:
        base=f"{prefix}_{itemid}"
        exprs.extend([
            f"AVG(CASE WHEN itemid={itemid} THEN valuenum END) {base}_mean",
            f"MIN(CASE WHEN itemid={itemid} THEN valuenum END) {base}_min",
            f"MAX(CASE WHEN itemid={itemid} THEN valuenum END) {base}_max",
            f"STDDEV_SAMP(CASE WHEN itemid={itemid} THEN valuenum END) {base}_std",
            f"SUM(CASE WHEN itemid={itemid} THEN 1 ELSE 0 END) {base}_count",
        ])
    con.execute(f"CREATE OR REPLACE TABLE {out_table} AS SELECT stay_id,{','.join(exprs)} FROM {event_table} GROUP BY stay_id")


def _build_inputs(con,root:Path,top_n:int,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE inputevents_24h AS
        SELECT
            c.stay_id,
            TRY_CAST(e.itemid AS BIGINT) itemid,
            TRY_CAST(e.amount AS DOUBLE) amount,
            TRY_CAST(e.rate AS DOUBLE) rate,
            TRY_CAST(e.patientweight AS DOUBLE) patientweight
        FROM {_read(root,'icu/inputevents.csv')} e
        JOIN cohort c ON TRY_CAST(e.stay_id AS BIGINT)=c.stay_id
        WHERE TRY_CAST(e.starttime AS TIMESTAMP)>=c.intime
            AND TRY_CAST(e.starttime AS TIMESTAMP)<c.intime+INTERVAL {hours} HOUR
            AND COALESCE(e.statusdescription,'') NOT IN ('Rewritten','Cancelled')
    """)
    con.execute("""
        CREATE OR REPLACE TABLE inputevents_item_counts AS
        SELECT itemid,COUNT(*) event_rows,COUNT(DISTINCT stay_id) stays
        FROM inputevents_24h
        GROUP BY itemid
        ORDER BY event_rows DESC
    """)
    itemids=[int(r[0]) for r in con.execute(f"SELECT itemid FROM inputevents_item_counts LIMIT {top_n}").fetchall()]
    exprs=["COUNT(*) input_total_events","COUNT(DISTINCT itemid) input_distinct_items","AVG(patientweight) input_patientweight_mean"]
    for itemid in itemids:
        base=f"input_{itemid}"
        exprs.extend([
            f"SUM(CASE WHEN itemid={itemid} THEN amount ELSE 0 END) {base}_amount_sum",
            f"AVG(CASE WHEN itemid={itemid} THEN rate END) {base}_rate_mean",
            f"SUM(CASE WHEN itemid={itemid} THEN 1 ELSE 0 END) {base}_count",
        ])
    con.execute(f"CREATE OR REPLACE TABLE inputevents_features AS SELECT stay_id,{','.join(exprs)} FROM inputevents_24h GROUP BY stay_id")


def _build_outputs(con,root:Path,top_n:int,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE outputevents_24h AS
        SELECT c.stay_id,TRY_CAST(e.itemid AS BIGINT) itemid,TRY_CAST(e.value AS DOUBLE) output_value
        FROM {_read(root,'icu/outputevents.csv')} e
        JOIN cohort c ON TRY_CAST(e.stay_id AS BIGINT)=c.stay_id
        WHERE TRY_CAST(e.value AS DOUBLE) IS NOT NULL
            AND TRY_CAST(e.charttime AS TIMESTAMP)>=c.intime
            AND TRY_CAST(e.charttime AS TIMESTAMP)<c.intime+INTERVAL {hours} HOUR
    """)
    con.execute("""
        CREATE OR REPLACE TABLE outputevents_item_counts AS
        SELECT itemid,COUNT(*) event_rows,COUNT(DISTINCT stay_id) stays
        FROM outputevents_24h GROUP BY itemid ORDER BY event_rows DESC
    """)
    itemids=[int(r[0]) for r in con.execute(f"SELECT itemid FROM outputevents_item_counts LIMIT {top_n}").fetchall()]
    exprs=["COUNT(*) output_total_events","COUNT(DISTINCT itemid) output_distinct_items"]
    for itemid in itemids:
        base=f"output_{itemid}"
        exprs.extend([
            f"SUM(CASE WHEN itemid={itemid} THEN output_value ELSE 0 END) {base}_value_sum",
            f"AVG(CASE WHEN itemid={itemid} THEN output_value END) {base}_value_mean",
            f"SUM(CASE WHEN itemid={itemid} THEN 1 ELSE 0 END) {base}_count",
        ])
    con.execute(f"CREATE OR REPLACE TABLE outputevents_features AS SELECT stay_id,{','.join(exprs)} FROM outputevents_24h GROUP BY stay_id")


def _build_procedure_events(con,root:Path,top_n:int,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE procedureevents_24h AS
        SELECT c.stay_id,TRY_CAST(e.itemid AS BIGINT) itemid,TRY_CAST(e.value AS DOUBLE) procedure_value
        FROM {_read(root,'icu/procedureevents.csv')} e
        JOIN cohort c ON TRY_CAST(e.stay_id AS BIGINT)=c.stay_id
        WHERE TRY_CAST(e.starttime AS TIMESTAMP)>=c.intime
            AND TRY_CAST(e.starttime AS TIMESTAMP)<c.intime+INTERVAL {hours} HOUR
            AND COALESCE(e.statusdescription,'') NOT IN ('Rewritten','Cancelled')
    """)
    con.execute("""
        CREATE OR REPLACE TABLE procedureevents_item_counts AS
        SELECT itemid,COUNT(*) event_rows,COUNT(DISTINCT stay_id) stays
        FROM procedureevents_24h GROUP BY itemid ORDER BY event_rows DESC
    """)
    itemids=[int(r[0]) for r in con.execute(f"SELECT itemid FROM procedureevents_item_counts LIMIT {top_n}").fetchall()]
    exprs=["COUNT(*) procedure_total_events","COUNT(DISTINCT itemid) procedure_distinct_items"]
    for itemid in itemids:
        exprs.append(f"SUM(CASE WHEN itemid={itemid} THEN 1 ELSE 0 END) procedure_{itemid}_count")
    con.execute(f"CREATE OR REPLACE TABLE procedureevents_features AS SELECT stay_id,{','.join(exprs)} FROM procedureevents_24h GROUP BY stay_id")


def _build_prescriptions(con,root:Path,top_n:int,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE prescriptions_24h AS
        SELECT c.stay_id,LOWER(TRIM(e.drug)) drug,LOWER(TRIM(e.route)) route
        FROM {_read(root,'hosp/prescriptions.csv')} e
        JOIN cohort c ON TRY_CAST(e.hadm_id AS BIGINT)=c.hadm_id
        WHERE TRY_CAST(e.starttime AS TIMESTAMP)>=c.intime
            AND TRY_CAST(e.starttime AS TIMESTAMP)<c.intime+INTERVAL {hours} HOUR
            AND e.drug IS NOT NULL
    """)
    con.execute("""
        CREATE OR REPLACE TABLE prescriptions_drug_counts AS
        SELECT drug,COUNT(*) event_rows,COUNT(DISTINCT stay_id) stays
        FROM prescriptions_24h
        GROUP BY drug
        ORDER BY event_rows DESC
    """)
    drugs=[r[0].replace("'","''") for r in con.execute(f"SELECT drug FROM prescriptions_drug_counts LIMIT {top_n}").fetchall()]
    exprs=["COUNT(*) rx_total_events","COUNT(DISTINCT drug) rx_distinct_drugs","COUNT(DISTINCT route) rx_distinct_routes"]
    for i,drug in enumerate(drugs):
        exprs.append(f"SUM(CASE WHEN drug='{drug}' THEN 1 ELSE 0 END) rx_top_{i}_count")
    con.execute(f"CREATE OR REPLACE TABLE prescriptions_features AS SELECT stay_id,{','.join(exprs)} FROM prescriptions_24h GROUP BY stay_id")


def _build_admin_counts(con,root:Path,hours:int)->None:
    con.execute(f"""
        CREATE OR REPLACE TABLE admin_features AS
        SELECT
            c.stay_id,
            COUNT(DISTINCT dx.icd_code) diagnosis_code_count,
            COUNT(DISTINCT pr.icd_code) procedure_icd_count,
            COUNT(DISTINCT s.curr_service) service_count,
            COUNT(DISTINCT t.transfer_id) transfer_count
        FROM cohort c
        LEFT JOIN {_read(root,'hosp/diagnoses_icd.csv')} dx ON TRY_CAST(dx.hadm_id AS BIGINT)=c.hadm_id
        LEFT JOIN {_read(root,'hosp/procedures_icd.csv')} pr ON TRY_CAST(pr.hadm_id AS BIGINT)=c.hadm_id
        LEFT JOIN {_read(root,'hosp/services.csv')} s ON TRY_CAST(s.hadm_id AS BIGINT)=c.hadm_id AND TRY_CAST(s.transfertime AS TIMESTAMP)<=c.intime+INTERVAL {hours} HOUR
        LEFT JOIN {_read(root,'hosp/transfers.csv')} t ON TRY_CAST(t.hadm_id AS BIGINT)=c.hadm_id AND TRY_CAST(t.intime AS TIMESTAMP)<=c.intime+INTERVAL {hours} HOUR
        GROUP BY c.stay_id
    """)
    con.execute("""
        CREATE OR REPLACE TABLE leakage_audit AS
        SELECT 'diagnosis_code_count' feature,'administrative diagnosis codes may be finalized after care; use as burden/EDA feature with caution' note
        UNION ALL SELECT 'procedure_icd_count','procedure ICD codes may include post-window procedures; count is kept separately for sensitivity analysis'
    """)


def _export_features(con,cfg:MimicConfig,root:Path)->None:
    feature_tables=[
        "chartevents_features","labevents_features","inputevents_features","outputevents_features",
        "procedureevents_features","prescriptions_features","admin_features",
    ]
    joins="\n".join([f"LEFT JOIN {t} USING(stay_id)" for t in feature_tables])
    con.execute(f"""
        CREATE OR REPLACE TABLE feature_base AS
        SELECT c.* EXCLUDE(outtime,edregtime,edouttime,last_careunit),{_feature_cols_sql(con,feature_tables)}
        FROM cohort c
        {joins}
    """)
    df=con.execute("SELECT * FROM feature_base ORDER BY row_id").fetchdf()
    df=df.rename(columns={"mortality_label":"label"})
    df["split"]=_client_splits(df,cfg.seed)
    exclude={
        "row_id","subject_id","hadm_id","stay_id","client_name","client_id","intime","admittime",
        "label","split","first_careunit",
    }
    LEAKAGE_COLS={"diagnosis_code_count","procedure_icd_count"}
    cat_cols=[c for c in ["admission_type","admission_location","insurance","language","marital_status","race","gender"] if c in df.columns]
    num_cols=[c for c in df.columns if c not in exclude and c not in cat_cols and c not in LEAKAGE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    cats=pd.get_dummies(df[cat_cols].fillna("missing"),prefix=cat_cols,dtype=np.float32) if cat_cols else pd.DataFrame(index=df.index)
    numeric=df[num_cols].apply(pd.to_numeric,errors="coerce").astype("float32")
    miss_rates=numeric.isna().mean()
    HIGH_MISS=0.95
    INDICATOR_THRESH=0.10
    dropped_cols=miss_rates[miss_rates>HIGH_MISS].index.tolist()
    indicator_cols=miss_rates[(miss_rates>INDICATOR_THRESH)&(miss_rates<=HIGH_MISS)].index.tolist()
    numeric=numeric.drop(columns=dropped_cols)
    indicators=pd.DataFrame({f"miss_{c}":df[c].isna().astype(np.float32) for c in indicator_cols},index=df.index) if indicator_cols else pd.DataFrame(index=df.index)
    train_mask=df["split"].eq("train").to_numpy()
    for c in numeric.columns:
        vals=numeric.loc[train_mask,c].dropna()
        if len(vals)<10:
            continue
        lo,hi=np.nanpercentile(vals,[1,99])
        numeric[c]=numeric[c].clip(lo,hi)
    features=pd.concat([numeric,indicators,cats],axis=1)
    imputer=SimpleImputer(strategy="median",keep_empty_features=True)
    scaler=StandardScaler()
    x_imp=imputer.fit_transform(features.loc[train_mask])
    scaler.fit(x_imp)
    x=scaler.transform(imputer.transform(features)).astype("float32")
    y=df["label"].astype("int64").to_numpy()
    client=df["client_id"].astype("int64").to_numpy()
    stay=df["stay_id"].astype("int64").to_numpy()
    feature_cols=features.columns.tolist()
    cleaning_log={
        "original_numeric_features":len(num_cols),
        "dropped_high_missingness":len(dropped_cols),
        "dropped_cols":dropped_cols,
        "missingness_indicators_added":len(indicator_cols),
        "indicator_cols":[f"miss_{c}" for c in indicator_cols],
        "leakage_cols_removed":sorted(LEAKAGE_COLS),
        "winsorize_percentiles":[1,99],
        "final_feature_count":len(feature_cols),
        "categorical_features":len(cats.columns),
        "numeric_features_after_drop":len(numeric.columns),
    }
    write_json(cfg.out/"preprocessing"/"cleaning_log.json",cleaning_log)
    matrix=pd.concat([df[["row_id","subject_id","hadm_id","stay_id","client_id","client_name","label","split"]].copy(),features.astype("float32")],axis=1)
    matrix.to_parquet(cfg.out/"preprocessing"/"feature_matrix_raw.parquet",index=False)
    pd.DataFrame(x,columns=feature_cols).assign(row_id=df["row_id"],stay_id=df["stay_id"],client_id=df["client_id"],label=y,split=df["split"]).to_parquet(cfg.out/"preprocessing"/"feature_matrix_scaled.parquet",index=False)
    np.savez_compressed(cfg.out/"preprocessing"/"model_arrays.npz",x=x,y=y,client_id=client,stay_id=stay,split=df["split"].to_numpy(),feature_names=np.array(feature_cols,dtype=object))
    with (cfg.out/"artifacts"/"mimic_preprocessor.pkl").open("wb") as f:
        pickle.dump({"imputer":imputer,"scaler":scaler,"feature_names":feature_cols,"categorical_columns":cat_cols,"numeric_columns":[c for c in num_cols if c not in dropped_cols],"dropped_cols":dropped_cols,"indicator_cols":indicator_cols,"leakage_removed":sorted(LEAKAGE_COLS)},f)
    write_json(cfg.out/"preprocessing"/"feature_columns.json",{"features":feature_cols,"numeric":[c for c in num_cols if c not in dropped_cols],"categorical":cat_cols,"indicators":[f"miss_{c}" for c in indicator_cols],"dropped":dropped_cols,"leakage_removed":sorted(LEAKAGE_COLS)})
    _export_tables(con,cfg.out)
    _write_split_summary(df,cfg.out)


def _feature_cols_sql(con,feature_tables:list[str])->str:
    cols=[]
    for table in feature_tables:
        table_cols=[r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
        cols.extend([f"{table}.{c}" for c in table_cols if c!="stay_id"])
    return ",".join(cols) if cols else "NULL no_event_features"


def _client_splits(df:pd.DataFrame,seed:int)->np.ndarray:
    split=np.array(["train"]*len(df),dtype=object)
    for cid,sub in df.groupby("client_id"):
        idx=sub.index.to_numpy()
        y=sub["label"].to_numpy()
        strat=y if len(np.unique(y))==2 and np.bincount(y.astype(int)).min()>=2 else None
        _,test=train_test_split(idx,test_size=0.25,random_state=seed+int(cid),stratify=strat)
        split[test]="test"
    return split


def _export_tables(con,out:Path)->None:
    for table,path in [
        ("client_map","preprocessing/client_map.csv"),
        ("chartevents_item_counts","preprocessing/chartevents_item_counts.csv"),
        ("labevents_item_counts","preprocessing/labevents_item_counts.csv"),
        ("inputevents_item_counts","preprocessing/inputevents_item_counts.csv"),
        ("outputevents_item_counts","preprocessing/outputevents_item_counts.csv"),
        ("procedureevents_item_counts","preprocessing/procedureevents_item_counts.csv"),
        ("prescriptions_drug_counts","preprocessing/prescriptions_drug_counts.csv"),
        ("leakage_audit","preprocessing/leakage_audit.csv"),
    ]:
        con.execute(f"COPY (SELECT * FROM {table}) TO '{str(out/path).replace("'","''")}' (HEADER,DELIMITER ',')")


def _write_split_summary(df:pd.DataFrame,out:Path)->None:
    rows=[]
    for (client_id,client_name,split),sub in df.groupby(["client_id","client_name","split"]):
        rows.append({
            "client_id":int(client_id),
            "client_name":client_name,
            "split":split,
            "rows":int(len(sub)),
            "mortality_rate":float(sub["label"].mean()),
            "deaths":int(sub["label"].sum()),
        })
    write_csv(out/"preprocessing"/"split_summary.csv",rows)


def _write_eda(con,cfg:MimicConfig)->None:
    df=pd.read_parquet(cfg.out/"preprocessing"/"feature_matrix_raw.parquet")
    write_csv(cfg.out/"eda"/"dataset_summary.csv",_dataset_summary(df,cfg))
    write_csv(cfg.out/"eda"/"client_summary.csv",_client_summary(df))
    write_csv(cfg.out/"eda"/"label_distribution.csv",_label_distribution(df))
    write_csv(cfg.out/"eda"/"feature_missingness.csv",_missingness(df))
    niid=_noniid(df)
    write_csv(cfg.out/"noniid"/"client_distribution_metrics.csv",niid)
    _plot_eda(df,cfg,niid)
    con.execute(f"COPY (SELECT * FROM cohort ORDER BY stay_id) TO '{str(cfg.out/'preprocessing'/'cohort.csv').replace("'","''")}' (HEADER,DELIMITER ',')")


def _dataset_summary(df:pd.DataFrame,cfg:MimicConfig)->list[dict]:
    feature_cols=[c for c in df.columns if c not in {"row_id","subject_id","hadm_id","stay_id","client_id","client_name","label","split"}]
    return [{
        "rows":int(len(df)),
        "features":len(feature_cols),
        "clients":int(df["client_id"].nunique()),
        "positive_label":int(df["label"].sum()),
        "negative_label":int((df["label"]==0).sum()),
        "mortality_rate":float(df["label"].mean()),
        "train_rows":int((df["split"]=="train").sum()),
        "test_rows":int((df["split"]=="test").sum()),
        "prediction_window_hours":cfg.hours,
    }]


def _client_summary(df:pd.DataFrame)->list[dict]:
    rows=[]
    for (cid,name),sub in df.groupby(["client_id","client_name"]):
        rows.append({
            "client_id":int(cid),
            "client_name":name,
            "rows":int(len(sub)),
            "mortality_rate":float(sub["label"].mean()),
            "deaths":int(sub["label"].sum()),
            "alive":int((sub["label"]==0).sum()),
            "train_rows":int((sub["split"]=="train").sum()),
            "test_rows":int((sub["split"]=="test").sum()),
        })
    return sorted(rows,key=lambda r:r["rows"],reverse=True)


def _label_distribution(df:pd.DataFrame)->list[dict]:
    return [
        {"label":0,"name":"survived","count":int((df["label"]==0).sum()),"proportion":float((df["label"]==0).mean())},
        {"label":1,"name":"hospital_expire_flag","count":int((df["label"]==1).sum()),"proportion":float((df["label"]==1).mean())},
    ]


def _missingness(df:pd.DataFrame)->list[dict]:
    keep={"row_id","subject_id","hadm_id","stay_id","client_id","client_name","label","split"}
    rows=[]
    for col in [c for c in df.columns if c not in keep]:
        miss=int(df[col].isna().sum())
        rows.append({"feature":col,"missing":miss,"missing_rate":float(miss/len(df))})
    return sorted(rows,key=lambda r:r["missing_rate"],reverse=True)


def _noniid(df:pd.DataFrame)->list[dict]:
    global_p=np.array([(df["label"]==0).mean(),(df["label"]==1).mean()])+1e-9
    global_p=global_p/global_p.sum()
    rows=[]
    for (cid,name),sub in df.groupby(["client_id","client_name"]):
        p=np.array([(sub["label"]==0).mean(),(sub["label"]==1).mean()])+1e-9
        p=p/p.sum()
        m=0.5*(p+global_p)
        rows.append({
            "client_id":int(cid),
            "client_name":name,
            "samples":int(len(sub)),
            "mortality_rate":float(p[1]),
            "label_entropy":float(-(p*np.log2(p)).sum()),
            "kl_divergence":float((p*np.log(p/global_p)).sum()),
            "js_divergence":float(0.5*(p*np.log(p/m)).sum()+0.5*(global_p*np.log(global_p/m)).sum()),
            "dominant_class_ratio":float(p.max()),
        })
    return sorted(rows,key=lambda r:r["js_divergence"],reverse=True)


def _plot_eda(df:pd.DataFrame,cfg:MimicConfig,niid:list[dict])->None:
    plot_dir=cfg.out/"plots"/"eda"
    _bar(_label_distribution(df),"name","count",plot_dir/"mortality_label_distribution.png","Mortality Label Distribution")
    _bar(_client_summary(df),"client_name","rows",plot_dir/"client_sample_counts.png","ICU Stays per Federated Client")
    _bar(_client_summary(df),"client_name","mortality_rate",plot_dir/"client_mortality_rates.png","Mortality Rate by ICU Client")
    _bar(niid,"client_name","js_divergence",plot_dir/"client_noniid_js_divergence.png","Client Label JS Divergence")
    _bar(niid,"client_name","kl_divergence",plot_dir/"client_noniid_kl_divergence.png","Client Label KL Divergence")
    _bar(niid,"client_name","label_entropy",plot_dir/"client_label_entropy.png","Client Label Entropy")
    _hist(df["anchor_age"].dropna(),plot_dir/"age_distribution.png","Age Distribution","anchor_age")
    _box_by_label(df,"anchor_age",plot_dir/"age_by_label.png","Age by Mortality Label")
    if "hours_hosp_before_icu" in df.columns:
        _hist(df["hours_hosp_before_icu"].dropna().clip(-100,500),plot_dir/"hours_hosp_before_icu_distribution.png","Hours in Hospital Before ICU","hours")
        _box_by_label(df,"hours_hosp_before_icu",plot_dir/"hours_hosp_before_icu_by_label.png","Hosp-to-ICU Hours by Mortality")
    if "observed_full_window" in df.columns:
        _bar([{"name":"full 24h","count":int((df["observed_full_window"]==1).sum())},{"name":"<24h","count":int((df["observed_full_window"]==0).sum())}],"name","count",plot_dir/"observation_window_completeness.png","Observation Window Completeness")
    _scatter_feature(df,"anchor_age","chartevents_total_events",plot_dir/"age_vs_chart_events.png","Age vs Chart Event Count","label")
    _scatter_feature(df,"chartevents_total_events","labevents_total_events",plot_dir/"chart_vs_lab_event_counts.png","Chart vs Lab Event Count","label")
    if "input_total_events" in df.columns:
        _scatter_feature(df,"chartevents_total_events","input_total_events",plot_dir/"chart_vs_input_events.png","Chart vs Input Event Count","label")
    if "rx_total_events" in df.columns:
        _hist(df["rx_total_events"].dropna(),plot_dir/"prescription_event_count_distribution.png","Prescription Event Count","rx_total_events")
    _feature_correlation(df,plot_dir/"feature_correlation_subset.png")
    _pca(df,plot_dir/"pca_by_mortality.png",plot_dir/"pca_3d_by_mortality.png","label")
    _pca(df,plot_dir/"pca_by_client.png",plot_dir/"pca_3d_by_client.png","client_name")
    top_missing=_missingness(df)[:30]
    _bar(top_missing,"feature","missing_rate",plot_dir/"top_missing_features.png","Top Missing Feature Rates")
    _stacked_mortality_by_client(df,plot_dir/"stacked_mortality_by_client.png")
    _client_sample_pie(df,plot_dir/"client_sample_pie.png")
    _violin_age_by_client(df,plot_dir/"violin_age_by_client.png")
    clean_log_path=cfg.out/"preprocessing"/"cleaning_log.json"
    if clean_log_path.exists():
        import json
        cl=json.loads(clean_log_path.read_text())
        _bar([
            {"stage":"Original numeric","count":cl.get("original_numeric_features",0)},
            {"stage":"Dropped (>95% miss)","count":cl.get("dropped_high_missingness",0)},
            {"stage":"Indicators added","count":cl.get("missingness_indicators_added",0)},
            {"stage":"Categorical","count":cl.get("categorical_features",0)},
            {"stage":"Final total","count":cl.get("final_feature_count",0)},
        ],"stage","count",plot_dir/"preprocessing_feature_pipeline.png","Feature Pipeline Summary")


def _stacked_mortality_by_client(df:pd.DataFrame,path:Path)->None:
    _prep(path)
    summary=df.groupby("client_name")["label"].agg(["sum","count"]).reset_index()
    summary.columns=["client","deaths","total"]
    summary["survived"]=summary["total"]-summary["deaths"]
    summary=summary.sort_values("total",ascending=True)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(summary["client"],summary["survived"],label="survived",color="#4CAF50")
    ax.barh(summary["client"],summary["deaths"],left=summary["survived"],label="expired",color="#F44336")
    ax.set_xlabel("ICU stays")
    ax.set_title("Mortality Breakdown by ICU Unit")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _client_sample_pie(df:pd.DataFrame,path:Path)->None:
    _prep(path)
    counts=df.groupby("client_name").size().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(7,7))
    ax.pie(counts,labels=counts.index,autopct="%1.1f%%",startangle=140,textprops={"fontsize":7})
    ax.set_title("Client Share of ICU Stays")
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _violin_age_by_client(df:pd.DataFrame,path:Path)->None:
    if "anchor_age" not in df.columns:
        return
    _prep(path)
    clients=sorted(df["client_name"].unique())
    data=[df.loc[df["client_name"]==c,"anchor_age"].dropna().values for c in clients]
    fig,ax=plt.subplots(figsize=(10,5))
    parts=ax.violinplot(data,showmeans=True,showmedians=True)
    ax.set_xticks(range(1,len(clients)+1))
    ax.set_xticklabels(clients,rotation=45,ha="right",fontsize=7)
    ax.set_ylabel("Age")
    ax.set_title("Age Distribution by ICU Unit")
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _prep(path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True)


def _bar(rows:list[dict],x:str,y:str,path:Path,title:str)->None:
    _prep(path)
    plt.figure(figsize=(9,4))
    plt.bar([str(r[x]) for r in rows],[float(r[y]) for r in rows])
    plt.xticks(rotation=45,ha="right")
    plt.title(title)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _hist(values:pd.Series,path:Path,title:str,xlabel:str)->None:
    _prep(path)
    plt.figure(figsize=(7,4))
    plt.hist(values,bins=30)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _box_by_label(df:pd.DataFrame,col:str,path:Path,title:str)->None:
    _prep(path)
    plt.figure(figsize=(6,4))
    data=[df.loc[df["label"]==0,col].dropna(),df.loc[df["label"]==1,col].dropna()]
    plt.boxplot(data,labels=["survived","expired"])
    plt.title(title)
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _scatter_feature(df:pd.DataFrame,x:str,y:str,path:Path,title:str,color:str)->None:
    if x not in df.columns or y not in df.columns:
        return
    _prep(path)
    sample=df.sample(min(len(df),8000),random_state=0)
    plt.figure(figsize=(6,4))
    sc=plt.scatter(sample[x],sample[y],c=sample[color].astype("category").cat.codes,s=8,alpha=0.45,cmap="viridis")
    plt.colorbar(sc,label=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path,dpi=160)
    plt.close()


def _feature_correlation(df:pd.DataFrame,path:Path)->None:
    keep={"row_id","subject_id","hadm_id","stay_id","client_id","label"}
    numeric=[c for c in df.columns if c not in keep and pd.api.types.is_numeric_dtype(df[c])]
    cols=numeric[:40]
    if len(cols)<2:
        return
    corr=df[cols].corr().fillna(0).to_numpy()
    _prep(path)
    plt.figure(figsize=(10,8))
    plt.imshow(corr,aspect="auto",cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(cols)),cols,rotation=90,fontsize=5)
    plt.yticks(range(len(cols)),cols,fontsize=5)
    plt.title("Feature Correlation Subset")
    plt.tight_layout()
    plt.savefig(path,dpi=180)
    plt.close()


def _pca(df:pd.DataFrame,path2d:Path,path3d:Path,label_col:str)->None:
    arr=pd.read_parquet(df.attrs.get("scaled_path","")) if False else None
    feature_cols=[c for c in df.columns if c not in {"row_id","subject_id","hadm_id","stay_id","client_id","client_name","label","split"} and pd.api.types.is_numeric_dtype(df[c])]
    x=df[feature_cols].apply(pd.to_numeric,errors="coerce").fillna(df[feature_cols].median(numeric_only=True)).fillna(0)
    sample=df.sample(min(len(df),5000),random_state=1)
    xs=x.loc[sample.index]
    pts=PCA(n_components=3,random_state=0).fit_transform(StandardScaler().fit_transform(xs))
    labels=sample[label_col].astype(str).to_numpy()
    uniq=sorted(set(labels))
    _prep(path2d)
    plt.figure(figsize=(7,5))
    for lab in uniq:
        mask=labels==lab
        plt.scatter(pts[mask,0],pts[mask,1],s=8,alpha=0.6,label=lab)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"PCA by {label_col}")
    if len(uniq)<=12:
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path2d,dpi=160)
    plt.close()
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(111,projection="3d")
    for lab in uniq:
        mask=labels==lab
        ax.scatter(pts[mask,0],pts[mask,1],pts[mask,2],s=8,alpha=0.6,label=lab)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"3D PCA by {label_col}")
    if len(uniq)<=12:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path3d,dpi=160)
    plt.close()
