import os
import zipfile
import tempfile
import hashlib
import requests
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="OULAD Insight Lab", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#0b1220;--panel:#111c2e;--panel2:#15243a;--line:#263750;--text:#edf3fc;--muted:#9aabc2;--accent:#4de0c1;--accent2:#7aa7ff}
.stApp{background:radial-gradient(ellipse at 10% 0%,#14283d 0%,#0b1220 44%,#080e19 100%);color:var(--text);font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:rgba(8,14,25,.72)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1b2d,#0b1423);border-right:1px solid var(--line)}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{font-size:1.05rem;letter-spacing:.04em}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#f5f8ff;letter-spacing:-.035em}
h1{font-size:2.35rem!important;font-weight:700!important;margin-bottom:.2rem}
h2{font-size:1.55rem!important}h3{font-size:1.12rem!important}
p,label,.stCaption{color:#c5d1e2}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(23,39,61,.96),rgba(15,27,44,.96));border:1px solid #2a3d58;padding:17px 18px;border-radius:16px;box-shadow:0 8px 22px rgba(0,0,0,.12)}
[data-testid="stMetricLabel"]{color:#a5b6cd!important;font-size:.82rem!important}
[data-testid="stMetricValue"]{color:#f4f8ff!important;font-weight:700}
[data-testid="stMetricDelta"]{font-size:.8rem}
.stButton button,.stDownloadButton button{background:linear-gradient(135deg,#1b3650,#14263b);color:#f5f8ff;border:1px solid #35516e;border-radius:11px;min-height:2.6rem;font-weight:600;transition:all .15s ease}
.stButton button:hover,.stDownloadButton button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px)}
.stButton button[kind="primary"]{background:linear-gradient(135deg,#1b8b81,#176d79);border:1px solid #49d9c0;color:white}
.stButton button[kind="primary"]:hover{background:linear-gradient(135deg,#24a99a,#1a8190);color:white}
[data-testid="stTabs"] button{font-weight:600}
[data-testid="stDataFrame"], [data-testid="stTable"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"]{background:rgba(17,28,46,.72);border:1px solid var(--line);border-radius:12px}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{background:#111e31;border-color:#30445f;border-radius:10px}
[data-testid="stFileUploader"]{background:rgba(17,28,46,.55);border:1px dashed #3a526f;border-radius:12px;padding:8px}
hr{border-color:#263750;margin:1.3rem 0}
.hero{padding:1.5rem 1.7rem;border:1px solid #2c4962;border-radius:20px;background:linear-gradient(115deg,rgba(23,56,74,.85),rgba(17,29,48,.9) 58%,rgba(28,43,70,.8));margin-bottom:1.25rem;box-shadow:0 16px 40px rgba(0,0,0,.15)}
.eyebrow{color:var(--accent);letter-spacing:.16em;font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:.6rem}
.hero-title{font-family:'Space Grotesk',sans-serif;font-size:2.3rem;line-height:1.1;font-weight:700;color:#f5f8ff;margin:0 0 .65rem 0}
.hero-copy{color:#b9c9dd;font-size:1rem;max-width:800px;line-height:1.65;margin:0}
.section-kicker{color:var(--accent);font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;font-weight:700;margin-bottom:.3rem}
.small-muted{color:#9aabc2;font-size:.88rem}
@media(max-width:700px){h1{font-size:1.8rem!important}.hero-title{font-size:1.8rem}.hero{padding:1.2rem}}
</style>
""", unsafe_allow_html=True)
DEFAULT = r"D:/Documents/Datawarehouse Project/Dataset"

def find_dir():
    for d in [os.environ.get("OULAD_DATA_DIR",""), str(Path(__file__).parent/"data"), DEFAULT, str(Path.cwd()/"Dataset"), str(Path.cwd())]:
        if d and (Path(d)/"studentInfo.csv").exists(): return d
    return DEFAULT

@st.cache_data(show_spinner=False)
def load_engagement(folder):
    p=Path(folder)
    students=pd.read_csv(p/"studentInfo.csv")
    vle=pd.read_csv(p/"studentVle.csv", usecols=["id_student","code_module","code_presentation","sum_click","date","id_site"])
    eng=vle.groupby(["id_student","code_module","code_presentation"],as_index=False).agg(
        total_clicks=("sum_click","sum"), active_days=("date","nunique"), unique_resources=("id_site","nunique"))
    return students.merge(eng,on=["id_student","code_module","code_presentation"],how="left").fillna(
        {"total_clicks":0,"active_days":0,"unique_resources":0})

def stylefig(fig,height=370):
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#c7d2e2",height=height,margin=dict(l=20,r=20,t=50,b=20))
    return fig

def score_silhouette(X, labels):
    labels=np.asarray(labels); mask=labels!=-1
    if len(set(labels[mask]))<2 or mask.sum()<3:return np.nan
    try:return silhouette_score(X[mask],labels[mask],sample_size=min(3000,mask.sum()),random_state=42)
    except:return np.nan

def clustering(folder):
    st.title("Student engagement clustering")
    st.caption("Explore engagement segments from aggregated student-module-presentation activity.")
    mode=st.radio("View mode",["Compare algorithms","View one algorithm"],horizontal=True)
    a,b,c=st.columns(3)
    features=a.multiselect("Features",["total_clicks","active_days","unique_resources"],["total_clicks","active_days","unique_resources"])
    k=b.slider("K for K-means / Hierarchical",2,10,4)
    sample=c.select_slider("Records to analyze",options=[1000,2500,5000,10000,20000,"All"],value=5000)
    if len(features)<2: st.warning("Select at least two features."); return
    df=load_engagement(folder)
    if sample!="All" and len(df)>sample: df=df.sample(int(sample),random_state=42)
    df=df.reset_index(drop=True)
    if mode=="View one algorithm":
        algo=st.selectbox("Algorithm",["K-means","Hierarchical","DBSCAN"])
        linkage=st.selectbox("Linkage",["ward","complete","average"]) if algo=="Hierarchical" else "ward"
        x,y=st.columns(2)
        eps=x.slider("DBSCAN epsilon",0.1,3.0,0.6,0.1) if algo=="DBSCAN" else 0.6
        minpts=y.slider("DBSCAN minimum points",3,50,10) if algo=="DBSCAN" else 10
        if st.button("Run clustering",type="primary"):
            X=StandardScaler().fit_transform(df[features].astype(float))
            if algo=="K-means": labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
            elif algo=="Hierarchical": labels=AgglomerativeClustering(n_clusters=k,linkage=linkage).fit_predict(X)
            else: labels=DBSCAN(eps=eps,min_samples=minpts,n_jobs=-1).fit_predict(X)
            sil=score_silhouette(X,labels)
            st.columns(3)[0].metric("Records",f"{len(df):,}")
            st.columns(3)[1].metric("Clusters",len(set(labels)-{-1}))
            st.columns(3)[2].metric("Silhouette",f"{sil:.3f}" if np.isfinite(sil) else "N/A")
            xy=PCA(n_components=2,random_state=42).fit_transform(X)
            plot=pd.DataFrame({"PC1":xy[:,0],"PC2":xy[:,1],"Cluster":labels.astype(str)})
            st.plotly_chart(stylefig(px.scatter(plot,x="PC1",y="PC2",color="Cluster",title=f"{algo} · PCA projection",opacity=.7)),use_container_width=True)
            df["cluster"]=labels
            st.dataframe(df.groupby("cluster")[features].mean().reset_index(),use_container_width=True)
            st.download_button("Download assignments",df.to_csv(index=False).encode(),"cluster_assignments.csv","text/csv")
    else:
        if st.button("Run algorithm comparison",type="primary"):
            X=StandardScaler().fit_transform(df[features].astype(float)); xy=PCA(n_components=2,random_state=42).fit_transform(X)
            rows=[]; plots=[]; profiles=[]
            for algo in ["K-means","Hierarchical","DBSCAN"]:
                if algo=="K-means": labels=KMeans(n_clusters=k,n_init=10,random_state=42).fit_predict(X)
                elif algo=="Hierarchical": labels=AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(X)
                else: labels=DBSCAN(eps=.6,min_samples=10,n_jobs=-1).fit_predict(X)
                sil=score_silhouette(X,labels)
                rows.append({"Algorithm":algo,"Clusters":len(set(labels)-{-1}),"Noise records":int(sum(labels==-1)),"Silhouette":sil})
                plots.append(pd.DataFrame({"PC1":xy[:,0],"PC2":xy[:,1],"Cluster":labels.astype(str),"Algorithm":algo}))
                tmp=df[features].copy();tmp["cluster"]=labels.astype(str);tmp["Algorithm"]=algo
                profiles.append(tmp.groupby(["Algorithm","cluster"])[features].mean().reset_index())
            res=pd.DataFrame(rows);st.dataframe(res,use_container_width=True,hide_index=True)
            st.plotly_chart(stylefig(px.bar(res.dropna(subset=["Silhouette"]),x="Algorithm",y="Silhouette",title="Silhouette comparison")),use_container_width=True)
            st.plotly_chart(stylefig(px.scatter(pd.concat(plots),x="PC1",y="PC2",color="Cluster",facet_col="Algorithm",title="PCA comparison",opacity=.65),520),use_container_width=True)
            st.subheader("Cluster engagement profiles");st.dataframe(pd.concat(profiles),use_container_width=True,hide_index=True)
            st.download_button("Download comparison",res.to_csv(index=False).encode(),"clustering_comparison.csv","text/csv")

def classifier(folder):
    st.title("Classification workbench")
    st.caption("Train selected algorithms on the same student-grouped holdout split.")
    chosen=st.multiselect("Algorithms",["Dummy baseline","Decision Tree","Random Forest","Extra Trees","Logistic Regression","KNN"],["Dummy baseline","Decision Tree","Random Forest"])
    testpct=st.slider("Test split (%)",10,35,20,5)
    trees=st.select_slider("Ensemble trees",options=[100,200,300,500],value=300)
    depth=st.selectbox("Tree maximum depth",["Unlimited",5,10,15,20])
    if st.button("Train and compare classifiers",type="primary"):
        # A newly trained model set invalidates any prediction made with the previous set.
        st.session_state.pop("new_student_prediction", None)
        df=load_engagement(folder)
        cols=["gender","region","highest_education","imd_band","age_band","num_of_prev_attempts","studied_credits","disability","code_module","code_presentation","total_clicks","active_days","unique_resources"]
        frame=df[cols+["final_result","id_student"]].copy().dropna(subset=["final_result"])
        y=frame.pop("final_result").astype(str);groups=frame.pop("id_student")
        for col in frame.select_dtypes(include="object").columns: frame[col]=frame[col].fillna("Unknown")
        frame=pd.get_dummies(frame,dummy_na=True)
        for col in frame.columns: frame[col]=pd.to_numeric(frame[col],errors="coerce").fillna(0)
        train,test=next(GroupShuffleSplit(n_splits=1,test_size=testpct/100,random_state=42).split(frame,y,groups))
        Xtr,Xte=frame.iloc[train],frame.iloc[test];ytr,yte=y.iloc[train],y.iloc[test]
        d=None if depth=="Unlimited" else int(depth); models={}; preds=[]; rows=[]
        for name in chosen:
            if name=="Dummy baseline": model=DummyClassifier(strategy="most_frequent")
            elif name=="Decision Tree": model=DecisionTreeClassifier(max_depth=d,min_samples_leaf=5,class_weight="balanced",random_state=42)
            elif name=="Random Forest": model=RandomForestClassifier(n_estimators=trees,max_depth=d,min_samples_leaf=5,class_weight="balanced_subsample",n_jobs=-1,random_state=42)
            elif name=="Extra Trees": model=ExtraTreesClassifier(n_estimators=trees,max_depth=d,min_samples_leaf=5,class_weight="balanced",n_jobs=-1,random_state=42)
            elif name=="Logistic Regression": model=LogisticRegression(max_iter=1000,class_weight="balanced")
            else: model=make_pipeline(StandardScaler(with_mean=False),KNeighborsClassifier(n_neighbors=5,weights="distance"))
            model.fit(Xtr,ytr); pred=model.predict(Xte);models[name]=model
            rows.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),"Macro-F1":f1_score(yte,pred,average="macro",zero_division=0),"Macro-Precision":__import__("sklearn.metrics").metrics.precision_score(yte,pred,average="macro",zero_division=0),"Macro-Recall":__import__("sklearn.metrics").metrics.recall_score(yte,pred,average="macro",zero_division=0)})
        res=pd.DataFrame(rows).sort_values("Macro-F1",ascending=False)
        st.dataframe(res.style.format({c:"{:.3f}" for c in ["Accuracy","Macro-F1","Macro-Precision","Macro-Recall"]}),use_container_width=True,hide_index=True)
        st.plotly_chart(stylefig(px.bar(res.melt(id_vars="Algorithm",value_vars=["Accuracy","Macro-F1"],var_name="Metric",value_name="Score"),x="Algorithm",y="Score",color="Metric",barmode="group",title="Classifier comparison")),use_container_width=True)
        st.session_state["classification_models"] = {
            "models": models, "columns": list(Xtr.columns), "features": cols,
            "classes": sorted(y.unique().tolist())
        }
        selected=st.selectbox("Confusion matrix model",list(models))
        pred=models[selected].predict(Xte); labels=["Distinction","Fail","Pass","Withdrawn"]
        cm=confusion_matrix(yte,pred,labels=labels)
        st.plotly_chart(stylefig(px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Teal",title=selected+" · confusion matrix",labels={"x":"Predicted","y":"Actual","color":"Count"})),use_container_width=True)
        if hasattr(models[selected],"feature_importances_"):
            imp=pd.DataFrame({"Feature":Xtr.columns,"Importance":models[selected].feature_importances_}).nlargest(15,"Importance")
            st.plotly_chart(stylefig(px.bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Feature importance")),use_container_width=True)
        st.download_button("Download comparison",res.to_csv(index=False).encode(),"classification_comparison.csv","text/csv")

    st.markdown("---")
    st.subheader("Predict a new student's final result")
    st.caption("Train the classifiers above first. You can enter values manually or generate a sample student profile.")
    saved = st.session_state.get("classification_models")
    if not saved:
        st.info("Choose algorithms and click **Train and compare classifiers** to enable new-student prediction.")
        return

    # The random profile button populates the same fields users can edit manually.
    categorical_options = {
        "gender": ["M", "F"],
        "region": ["East Anglian Region", "East Midlands Region", "Ireland", "London Region", "North Region", "North Western Region", "Scotland", "South East Region", "South Region", "South West Region", "West Midlands Region", "Wales", "Yorkshire Region"],
        "highest_education": ["No Formal quals", "Lower Than A Level", "A Level or Equivalent", "HE Qualification", "Post Graduate Qualification"],
        "imd_band": ["0-10%", "10-20", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%", "Unknown"],
        "age_band": ["0-35", "35-55", "55<="],
        "disability": ["N", "Y"],
        "code_module": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"],
        "code_presentation": ["2013J", "2014J", "2013B", "2014B"],
    }
    numeric_options = {
        "num_of_prev_attempts": (0, 3), "studied_credits": (30, 120),
        "total_clicks": (0, 8000), "active_days": (0, 240), "unique_resources": (0, 300),
    }
    if st.button("🎲 Fill random student values", key="random_student_values"):
        rng = np.random.default_rng()
        for col, options in categorical_options.items():
            st.session_state["newstudent_" + col] = str(rng.choice(options[:-1] if col == "imd_band" else options))
        for col, (lo, hi) in numeric_options.items():
            st.session_state["newstudent_" + col] = int(rng.integers(lo, hi + 1))
        st.session_state.pop("new_student_prediction", None)

    with st.form("new_student_prediction_form"):
        st.markdown("**Student background**")
        c1, c2, c3 = st.columns(3)
        inputs = {}
        for i, col in enumerate(["gender", "region", "highest_education", "imd_band", "age_band", "disability"]):
            with [c1, c2, c3][i % 3]:
                opts = categorical_options[col]
                default = st.session_state.get("newstudent_" + col, "Unknown" if col == "imd_band" else opts[0])
                if default not in opts: default = opts[0]
                inputs[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(default), key="newstudent_" + col)
        st.markdown("**Course and engagement**")
        c1, c2, c3 = st.columns(3)
        for i, col in enumerate(["code_module", "code_presentation"]):
            with [c1, c2][i]:
                opts = categorical_options[col]
                default = st.session_state.get("newstudent_" + col, opts[0])
                if default not in opts: default = opts[0]
                inputs[col] = st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(default), key="newstudent_" + col)
        for i, col in enumerate(numeric_options):
            with [c1, c2, c3][i % 3]:
                lo, hi = numeric_options[col]
                default = int(st.session_state.get("newstudent_" + col, lo))
                inputs[col] = st.number_input(col.replace("_", " ").title(), min_value=lo, max_value=hi, value=default, step=1, key="newstudent_" + col)
        model_name = st.selectbox("Prediction model", list(saved["models"]), index=(list(saved["models"]).index("Random Forest") if "Random Forest" in saved["models"] else 0), key="newstudent_model")
        submitted = st.form_submit_button("Predict final result", type="primary")
    if submitted:
        new_df = pd.DataFrame([inputs])
        for col in new_df.select_dtypes(include="object").columns:
            new_df[col] = new_df[col].fillna("Unknown")
        new_df = pd.get_dummies(new_df, dummy_na=True)
        new_df = new_df.reindex(columns=saved["columns"], fill_value=0)
        model = saved["models"][model_name]
        result = str(model.predict(new_df)[0])
        st.session_state["new_student_prediction"] = {
            "result": result, "model": model_name, "input": new_df.to_dict(orient="records")[0]
        }
    outcome = st.session_state.get("new_student_prediction")
    if outcome:
        st.success(f"Predicted final result: **{outcome['result']}**")
        st.caption(f"Model used: {outcome['model']}. This is a model estimate, not a guarantee.")
        # Only render a stored result if its model still exists in the current trained set.
        if outcome["model"] in saved["models"]:
            model = saved["models"][outcome["model"]]
            if hasattr(model, "predict_proba"):
                saved_input = pd.DataFrame([outcome["input"]]).reindex(columns=saved["columns"], fill_value=0)
                probs = model.predict_proba(saved_input)
                prob_df = pd.DataFrame({
                    "Class": model.classes_, "Estimated probability": probs[0]
                }).sort_values("Estimated probability", ascending=False)
                st.dataframe(
                    prob_df.style.format({"Estimated probability": "{:.1%}"}),
                    hide_index=True, use_container_width=True
                )
        else:
            st.info("The classifier set has changed. Please predict again using the newly trained models.")

def rules(folder):
    st.title("Association rule mining")
    st.caption("Mine activity-type co-occurrence patterns from student VLE transactions.")
    p=Path(folder)
    if not (p/"studentVle.csv").exists() or not (p/"vle.csv").exists():
        st.warning("Required files: studentVle.csv and vle.csv");return
    a,b,c=st.columns(3)
    support=a.slider("Minimum support (%)",1,20,5)
    confidence=b.slider("Minimum confidence (%)",10,100,60,5)
    lift=c.slider("Minimum lift",1.0,10.0,1.0,.5)
    limit=st.select_slider("Maximum displayed rules",options=[25,50,100,250],value=50)
    st.info("Rules are recalculated after you click Run. Higher support thresholds usually use less memory.")
    if st.button("Mine rules",type="primary"):
        from itertools import combinations
        with st.spinner("Reading activity logs and mining rules..."):
            try:
                vle=pd.read_csv(p/"studentVle.csv",usecols=["id_student","code_module","code_presentation","id_site"])
                sites=pd.read_csv(p/"vle.csv",usecols=["id_site","activity_type"])
                joined=vle.merge(sites,on="id_site",how="left")
                tx=joined.groupby(["id_student","code_module","code_presentation"])["activity_type"].apply(lambda s:set(s.dropna().astype(str))).tolist()
                n=len(tx); counts={}
                for items in tx:
                    for r in range(1,min(4,len(items)+1)):
                        for combo in combinations(sorted(items),r): counts[combo]=counts.get(combo,0)+1
                rows=[]; mincount=max(1,int(np.ceil(support/100*n)))
                for itemset,count in counts.items():
                    if len(itemset)<2 or count<mincount:continue
                    for size in range(1,len(itemset)):
                        for lhs in combinations(itemset,size):
                            rhs=tuple(x for x in itemset if x not in lhs)
                            lc=counts.get(tuple(sorted(lhs)),0);rc=counts.get(tuple(sorted(rhs)),0)
                            if lc and rc:
                                conf=count/lc; liftval=conf/(rc/n)
                                if conf>=confidence/100 and liftval>=lift:
                                    rows.append({"Antecedent":" + ".join(lhs),"Consequent":" + ".join(rhs),"Support":count/n,"Confidence":conf,"Lift":liftval,"Count":count})
                res=pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["Antecedent","Consequent","Support","Confidence","Lift","Count"])
                if len(res):
                    res=res.sort_values(["Lift","Confidence"],ascending=False).head(limit)
                    st.success(f"{len(res)} matching rules shown from {n:,} transactions.")
                    st.dataframe(res.style.format({"Support":"{:.2%}","Confidence":"{:.2%}","Lift":"{:.2f}"}),use_container_width=True,hide_index=True)
                    st.plotly_chart(stylefig(px.scatter(res,x="Support",y="Confidence",size="Lift",color="Lift",hover_data=["Antecedent","Consequent"],title="Support vs confidence")),use_container_width=True)
                else:st.warning("No rules matched. Lower support, confidence or lift.")
                st.download_button("Download rules CSV",res.to_csv(index=False).encode(),"association_rules.csv","text/csv")
            except Exception as e:st.error(f"Rule mining error: {e}")

def regression_frame(folder):
    p=Path(folder)
    assessments=pd.read_csv(p/"assessments.csv")
    scores=pd.read_csv(p/"studentAssessment.csv")
    info=pd.read_csv(p/"studentInfo.csv")
    scores=scores.merge(assessments[["id_assessment","code_module","code_presentation","assessment_type","weight"]],on="id_assessment",how="left")
    cols=["gender","region","highest_education","age_band","disability","num_of_prev_attempts","studied_credits"]
    scores=scores.merge(info[["id_student","code_module","code_presentation"]+cols],on=["id_student","code_module","code_presentation"],how="left").dropna(subset=["score"])
    feats=["assessment_type","weight"]+cols+["code_module","code_presentation"]
    frame=scores[feats+["score","id_student"]].copy()
    for col in frame.select_dtypes(include="object").columns:frame[col]=frame[col].fillna("Unknown")
    frame=pd.get_dummies(frame,columns=frame.select_dtypes(include="object").columns.tolist(),dummy_na=True)
    for col in frame.columns:
        if col not in ["score","id_student"]:frame[col]=pd.to_numeric(frame[col],errors="coerce").fillna(0)
    return frame

def prediction(folder):
    st.title("Assessment score prediction")
    st.caption("Compare regression methods for assessment score (0–100).")
    methods=st.multiselect("Regression methods",["Mean baseline","Linear Regression","Ridge Regression","Decision Tree","Random Forest","Extra Trees"],["Mean baseline","Linear Regression","Decision Tree","Random Forest"])
    trees=st.select_slider("Ensemble trees",options=[100,200,300],value=200)
    depth=st.selectbox("Tree maximum depth",["Unlimited",5,10,15,20])
    testpct=st.slider("Test split (%)",10,30,20,5)
    if st.button("Train and compare regressors",type="primary"):
        try:
            frame=regression_frame(folder);y=frame.pop("score").astype(float);groups=frame.pop("id_student")
            train,test=next(GroupShuffleSplit(n_splits=1,test_size=testpct/100,random_state=42).split(frame,y,groups))
            Xtr,Xte=frame.iloc[train],frame.iloc[test];ytr,yte=y.iloc[train],y.iloc[test]
            d=None if depth=="Unlimited" else int(depth);preds={};rows=[]
            for name in methods:
                if name=="Mean baseline":pred=np.repeat(ytr.mean(),len(yte))
                else:
                    if name=="Linear Regression":model=LinearRegression()
                    elif name=="Ridge Regression":model=Ridge(alpha=1.0)
                    elif name=="Decision Tree":model=DecisionTreeRegressor(max_depth=d,min_samples_leaf=5,random_state=42)
                    elif name=="Random Forest":model=RandomForestRegressor(n_estimators=trees,max_depth=d,min_samples_leaf=5,n_jobs=-1,random_state=42)
                    else:model=ExtraTreesRegressor(n_estimators=trees,max_depth=d,min_samples_leaf=5,n_jobs=-1,random_state=42)
                    model.fit(Xtr,ytr);pred=model.predict(Xte)
                pred=np.clip(pred,0,100);preds[name]=pred
                rows.append({"Method":name,"MAE":mean_absolute_error(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"R²":r2_score(yte,pred)})
            st.session_state["regression_results"]={"results":pd.DataFrame(rows).sort_values("RMSE"),"predictions":preds,"actual":yte.to_numpy()}
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # Keep trained predictions in session state so changing the graph selector
    # reruns the page without losing the model outputs.
    saved=st.session_state.get("regression_results")
    if saved:
        res=saved["results"]
        preds=saved["predictions"]
        st.dataframe(res.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","R²":"{:.3f}"}),use_container_width=True,hide_index=True)
        st.plotly_chart(stylefig(px.bar(res.melt(id_vars="Method",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error"),x="Method",y="Error",color="Metric",barmode="group",title="Regression error comparison")),use_container_width=True)
        selected=st.selectbox("Actual vs predicted",list(preds),key="regression_actual_predicted_model")
        sc=pd.DataFrame({"Actual":saved["actual"],"Predicted":preds[selected]})
        fig=px.scatter(sc,x="Actual",y="Predicted",opacity=.45,title=selected+" · actual vs predicted")
        fig.add_trace(go.Scatter(x=[0,100],y=[0,100],mode="lines",name="Ideal",line=dict(dash="dash",color="#42d6c4")))
        st.plotly_chart(stylefig(fig),use_container_width=True)
        st.download_button("Download comparison",res.to_csv(index=False).encode(),"regression_comparison.csv","text/csv")

def main():
    st.markdown("""
    <style>
    .courseiq-brand{display:flex;align-items:center;gap:15px;margin:0 0 .5rem 0}
    .courseiq-monogram{width:58px;height:58px;flex:0 0 58px;border-radius:15px;
      display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0d233a,#173b50);
      border:1px solid #28536a;color:#38d7d0;font-size:25px;font-weight:850;letter-spacing:-3px}
    .courseiq-monogram span{color:#38d7d0}
    .courseiq-title{font-size:2.05rem;line-height:1.05;font-weight:800;letter-spacing:-.04em;color:#f4f8fc}
    .courseiq-by{font-size:.83rem;color:#a9b9cb;margin-top:5px;letter-spacing:.04em}
    .courseiq-subtitle{color:#a9b9cb;font-size:.93rem;margin:.25rem 0 1.15rem 0}
    .courseiq-nav-label{font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;color:#7f96ad;font-weight:700;margin:.5rem 0 .35rem 0}
    div[data-testid="stRadio"] > div[role="radiogroup"]{gap:.45rem;flex-wrap:wrap}
    div[data-testid="stRadio"] > div[role="radiogroup"] label{background:#111d2d;border:1px solid #263a52;border-radius:10px;padding:.48rem .85rem;margin:0}
    div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked){background:#123b50;border-color:#25c9c7;color:#f5ffff}
    div[data-testid="stRadio"] > div[role="radiogroup"] label p{font-weight:600;font-size:.88rem}
    .team-footer{margin-top:3rem;padding:1.2rem 1rem;border-top:1px solid #263750;text-align:center;color:#9aabc2;font-size:.88rem}
    .team-footer .names{color:#edf3fc;font-weight:600;margin-top:.35rem;letter-spacing:.02em}
    </style>
    <div class="courseiq-brand">
      <div class="courseiq-monogram"><span>R</span><span>H</span><span>S</span></div>
      <div><div class="courseiq-title">CourseIQ</div><div class="courseiq-by">by Mining Minds</div></div>
    </div>
    <div class="courseiq-subtitle">Data-Driven Course Recommendation System · OULAD Dataset</div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # Automatically load repository data in Streamlit Cloud; local/upload modes remain available.
    @st.cache_data(show_spinner=False, max_entries=2)
    def download_github_file(filename, token):
        url = f"https://raw.githubusercontent.com/rithishk2506/oulad-insight-lab/main/{filename}"
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw+json"},
            timeout=(20, 300),
        )
        response.raise_for_status()
        return response.content

    def prepare_github_dataset():
        token = st.secrets.get("GITHUB_TOKEN", "")
        if not token:
            st.sidebar.error("GITHUB_TOKEN is missing. Add it in Streamlit app Settings → Secrets.")
            return ""
        # Keep the extracted dataset in the app's temporary directory for this running instance.
        data_dir = Path(tempfile.gettempdir()) / "oulad_insight_lab_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        required_files = [
            "studentInfo.csv", "vle.csv", "assessments.csv", "studentAssessment.csv"
        ]
        # Download smaller files only when not already present.
        for filename in required_files:
            target = data_dir / filename
            if not target.exists() or target.stat().st_size == 0:
                try:
                    target.write_bytes(download_github_file(filename, token))
                except Exception as exc:
                    st.sidebar.error(f"Could not download {filename} from the private GitHub repository: {exc}")
                    return ""
        student_vle = data_dir / "studentVle.csv"
        if not student_vle.exists() or student_vle.stat().st_size == 0:
            try:
                zip_path = data_dir / "studentVle.zip"
                if not zip_path.exists() or zip_path.stat().st_size == 0:
                    with st.sidebar.status("Downloading studentVle.zip from private GitHub repository…", expanded=True):
                        zip_path.write_bytes(download_github_file("studentVle.zip", token))
                with st.sidebar.status("Extracting studentVle.csv…", expanded=True):
                    with zipfile.ZipFile(zip_path) as archive:
                        member = next(
                            (n for n in archive.namelist() if Path(n).name.lower() == "studentvle.csv"),
                            None,
                        )
                        if not member:
                            raise FileNotFoundError("studentVle.csv was not found inside studentVle.zip.")
                        with archive.open(member) as source, student_vle.open("wb") as dest:
                            while True:
                                chunk = source.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                dest.write(chunk)
            except Exception as exc:
                st.sidebar.error(f"Could not download/extract studentVle.zip: {exc}")
                return ""
        return str(data_dir)

    source = st.sidebar.radio(
        "Dataset source",
        ["Automatic (private GitHub)", "Local folder", "Upload CSV files"],
        index=0,
    )
    if source == "Automatic (private GitHub)":
        folder = prepare_github_dataset()
        if folder:
            st.sidebar.success("Dataset loaded from your private GitHub repository.")
            st.sidebar.caption("The extracted studentVle.csv is large; first load may take several minutes and needs sufficient disk space.")
    elif source == "Local folder":
        folder = st.sidebar.text_input("Dataset folder", find_dir())
    else:
        st.sidebar.caption(
            "Upload the smaller CSV files plus studentVle.zip. "
            "The ZIP must contain studentVle.csv."
        )
        uploaded = st.sidebar.file_uploader(
            "Choose OULAD CSV files and studentVle.zip",
            type=["csv", "zip"],
            accept_multiple_files=True,
            help="Required: studentInfo.csv, studentVle.zip (containing studentVle.csv), "
                 "vle.csv, assessments.csv, studentAssessment.csv"
        )
        required_names = ["studentInfo.csv", "studentVle.csv", "vle.csv", "assessments.csv", "studentAssessment.csv"]
        upload_map = {f.name: f for f in uploaded}
        zip_uploads = [f for f in uploaded if f.name.lower().endswith(".zip")]
        student_vle_upload = upload_map.get("studentVle.csv")
        student_vle_zip = None
        for zf in zip_uploads:
            try:
                with zipfile.ZipFile(zf) as archive:
                    if any(Path(n).name.lower() == "studentvle.csv" for n in archive.namelist()):
                        student_vle_zip = zf
                        break
            except zipfile.BadZipFile:
                continue
        missing_uploads = [name for name in required_names if name != "studentVle.csv" and name not in upload_map]
        if student_vle_upload is None and student_vle_zip is None:
            missing_uploads.insert(1, "studentVle.zip (containing studentVle.csv)")
        if missing_uploads:
            st.sidebar.warning("Still needed: " + ", ".join(missing_uploads))
            folder = ""
        else:
            upload_sizes = [f"{name}:{upload_map[name].size}" for name in required_names if name in upload_map]
            if student_vle_zip is not None:
                upload_sizes.append(f"studentVle.zip:{student_vle_zip.size}")
            upload_signature = hashlib.md5("|".join(upload_sizes).encode()).hexdigest()
            folder = str(Path(tempfile.gettempdir()) / ("oulad_upload_" + upload_signature))
            data_dir = Path(folder)
            data_dir.mkdir(parents=True, exist_ok=True)
            for name in required_names:
                target = data_dir / name
                if name == "studentVle.csv" and student_vle_upload is None:
                    if not target.exists():
                        try:
                            with zipfile.ZipFile(student_vle_zip) as archive:
                                member = next(n for n in archive.namelist() if Path(n).name.lower() == "studentvle.csv")
                                with archive.open(member) as source, target.open("wb") as dest:
                                    while True:
                                        chunk = source.read(8 * 1024 * 1024)
                                        if not chunk:
                                            break
                                        dest.write(chunk)
                        except Exception as exc:
                            st.sidebar.error(f"Could not extract studentVle.csv from ZIP: {exc}")
                            folder = ""
                            break
                elif name in upload_map:
                    if not target.exists() or target.stat().st_size != upload_map[name].size:
                        target.write_bytes(upload_map[name].getvalue())
            if folder:
                st.sidebar.success("All five required files are available for this session.")
                st.sidebar.warning("studentVle.csv expands to several GB; Streamlit Cloud may run out of disk space or memory.")
    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()
    st.markdown('<div class="courseiq-nav-label">Explore CourseIQ</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigate to",
        ["Overview", "Clustering", "Classification", "Association Rules", "Prediction"],
        horizontal=True,
        label_visibility="collapsed",
        key="courseiq_top_navigation",
    )
    st.markdown("---")
    if page=="Overview":
        st.markdown("<div class='hero'><div class='eyebrow'>STUDENT LEARNING ANALYTICS · PROJECT WORKSPACE</div><div class='hero-title'>OULAD Insight Lab</div><p class='hero-copy'>Turn virtual learning activity into interpretable insights. Explore engagement segments, compare classification models, discover activity patterns, and estimate assessment performance.</p></div>",unsafe_allow_html=True)
        required=["studentInfo.csv","studentVle.csv","vle.csv","assessments.csv","studentAssessment.csv"]
        p=Path(folder) if folder else None
        found=sum((p/f).exists() for f in required) if p else 0
        c=st.columns(4)
        c[0].metric("Dataset readiness",f"{found}/{len(required)} files")
        c[1].metric("Analytics modules","4")
        c[2].metric("Clustering algorithms","3")
        c[3].metric("Workspace status","Ready" if found==5 else "Setup needed")
        st.markdown("---")
        left,right=st.columns([1.3,1])
        with left:
            st.markdown("<div class='section-kicker'>EXPLORE THE WORKSPACE</div>",unsafe_allow_html=True)
            st.subheader("Analytics modules")
            st.write("**01 · Clustering**  — K-means, Hierarchical and DBSCAN engagement segments")
            st.write("**02 · Classification**  — Dummy baseline, KNN, Decision Tree, Random Forest, Extra Trees and Logistic Regression")
            st.write("**03 · Association rules**  — Co-occurrence patterns using support, confidence and lift")
            st.write("**04 · Score prediction**  — Baseline and regression models for assessment scores")
        with right:
            st.markdown("<div class='section-kicker'>DATA STATUS</div>",unsafe_allow_html=True)
            st.subheader("Dataset readiness")
            for f in required:st.write(("🟢 " if p and (p/f).exists() else "🟠 ")+f)
            if found<5:
                st.warning("Choose a valid local dataset folder or upload all five required CSV files in the sidebar.")
        st.info("Results are computed at runtime. Metrics can differ from earlier exports when preprocessing, model settings, or train/test splits change.")
        st.caption("OULAD Insight Lab · Data warehousing and data mining project")
    elif not folder:
        st.title("Dataset required")
        st.info("Use the sidebar to select a dataset folder or upload all five required OULAD CSV files.")
    elif page=="Clustering":clustering(folder)
    elif page=="Classification":classifier(folder)
    elif page=="Association Rules":rules(folder)
    else:prediction(folder)


    st.markdown("""
    <div class="team-footer">
      <div>CourseIQ · Data Warehousing and Data Mining Project</div>
      <div class="names">Rithish K · Hemanathan R · Sarika Vadakkepat</div>
      <div style="font-size:.76rem;margin-top:.35rem;color:#7f96ad">by Mining Minds</div>
    </div>
    """, unsafe_allow_html=True)

if __name__=="__main__":main()
