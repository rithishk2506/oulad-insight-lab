import os
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

st.set_page_config(page_title="OULAD Insight Lab", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:#08111f;color:#e6edf7;font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"]{background:#0d192b;border-right:1px solid #20324a}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;color:#f4f8ff;letter-spacing:-.03em}
p,label{color:#c7d2e2}
div[data-testid="stMetric"]{background:#101e32;border:1px solid #233650;padding:16px;border-radius:13px}
div[data-testid="stMetricLabel"]{color:#91a7c0} div[data-testid="stMetricValue"]{color:#f4f8ff}
.stButton button,.stDownloadButton button{background:#142840;color:#f4f8ff;border:1px solid #2c4564;border-radius:9px}
.stButton button:hover{border-color:#42d6c4;color:#42d6c4}
hr{border-color:#20324a}
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
    chosen=st.multiselect("Algorithms",["Dummy baseline","Decision Tree","Random Forest","Extra Trees","Logistic Regression"],["Dummy baseline","Decision Tree","Random Forest"])
    testpct=st.slider("Test split (%)",10,35,20,5)
    trees=st.select_slider("Ensemble trees",options=[100,200,300,500],value=300)
    depth=st.selectbox("Tree maximum depth",["Unlimited",5,10,15,20])
    if st.button("Train and compare classifiers",type="primary"):
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
            else: model=LogisticRegression(max_iter=1000,class_weight="balanced")
            model.fit(Xtr,ytr); pred=model.predict(Xte);models[name]=model
            rows.append({"Algorithm":name,"Accuracy":accuracy_score(yte,pred),"Macro-F1":f1_score(yte,pred,average="macro",zero_division=0),"Macro-Precision":__import__("sklearn.metrics").metrics.precision_score(yte,pred,average="macro",zero_division=0),"Macro-Recall":__import__("sklearn.metrics").metrics.recall_score(yte,pred,average="macro",zero_division=0)})
        res=pd.DataFrame(rows).sort_values("Macro-F1",ascending=False)
        st.dataframe(res.style.format({c:"{:.3f}" for c in ["Accuracy","Macro-F1","Macro-Precision","Macro-Recall"]}),use_container_width=True,hide_index=True)
        st.plotly_chart(stylefig(px.bar(res.melt(id_vars="Algorithm",value_vars=["Accuracy","Macro-F1"],var_name="Metric",value_name="Score"),x="Algorithm",y="Score",color="Metric",barmode="group",title="Classifier comparison")),use_container_width=True)
        selected=st.selectbox("Confusion matrix model",list(models))
        pred=models[selected].predict(Xte); labels=["Distinction","Fail","Pass","Withdrawn"]
        cm=confusion_matrix(yte,pred,labels=labels)
        st.plotly_chart(stylefig(px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Teal",title=selected+" · confusion matrix",labels={"x":"Predicted","y":"Actual","color":"Count"})),use_container_width=True)
        if hasattr(models[selected],"feature_importances_"):
            imp=pd.DataFrame({"Feature":Xtr.columns,"Importance":models[selected].feature_importances_}).nlargest(15,"Importance")
            st.plotly_chart(stylefig(px.bar(imp.sort_values("Importance"),x="Importance",y="Feature",orientation="h",title="Feature importance")),use_container_width=True)
        st.download_button("Download comparison",res.to_csv(index=False).encode(),"classification_comparison.csv","text/csv")

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
            res=pd.DataFrame(rows).sort_values("RMSE")
            st.dataframe(res.style.format({"MAE":"{:.2f}","RMSE":"{:.2f}","R²":"{:.3f}"}),use_container_width=True,hide_index=True)
            st.plotly_chart(stylefig(px.bar(res.melt(id_vars="Method",value_vars=["MAE","RMSE"],var_name="Metric",value_name="Error"),x="Method",y="Error",color="Metric",barmode="group",title="Regression error comparison")),use_container_width=True)
            selected=st.selectbox("Actual vs predicted",list(preds))
            sc=pd.DataFrame({"Actual":yte.values,"Predicted":preds[selected]})
            fig=px.scatter(sc,x="Actual",y="Predicted",opacity=.45,title=selected+" · actual vs predicted")
            fig.add_trace(go.Scatter(x=[0,100],y=[0,100],mode="lines",name="Ideal",line=dict(dash="dash",color="#42d6c4")))
            st.plotly_chart(stylefig(fig),use_container_width=True)
            st.download_button("Download comparison",res.to_csv(index=False).encode(),"regression_comparison.csv","text/csv")
        except Exception as e:st.error(f"Prediction error: {e}")

def main():
    st.sidebar.markdown("### ◈ OULAD INSIGHT LAB")
    st.sidebar.caption("DATA MINING · DATA WAREHOUSING")

    # Cloud-compatible dataset selection. Local folder mode is retained for desktop use.
    source = st.sidebar.radio("Dataset source", ["Local folder", "Upload CSV files"])
    if source == "Local folder":
        folder = st.sidebar.text_input("Dataset folder", find_dir())
    else:
        st.sidebar.caption("Upload the five OULAD CSV files. Large files may exceed the hosting upload limit.")
        uploaded = st.sidebar.file_uploader(
            "Choose OULAD CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Required: studentInfo.csv, studentVle.csv, vle.csv, assessments.csv, studentAssessment.csv"
        )
        required_names = ["studentInfo.csv", "studentVle.csv", "vle.csv", "assessments.csv", "studentAssessment.csv"]
        upload_map = {f.name: f for f in uploaded}
        missing_uploads = [name for name in required_names if name not in upload_map]
        if missing_uploads:
            st.sidebar.warning("Still needed: " + ", ".join(missing_uploads))
            folder = ""
        else:
            import tempfile
            import hashlib
            upload_signature = hashlib.md5(
                "|".join(f"{name}:{upload_map[name].size}" for name in required_names).encode()
            ).hexdigest()
            folder = str(Path(tempfile.gettempdir()) / ("oulad_upload_" + upload_signature))
            Path(folder).mkdir(parents=True, exist_ok=True)
            for name in required_names:
                target = Path(folder) / name
                if not target.exists() or target.stat().st_size != upload_map[name].size:
                    target.write_bytes(upload_map[name].getvalue())
            st.sidebar.success("All five required files are available for this session.")

    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()
    page=st.sidebar.radio("NAVIGATION",["Overview","Clustering","Classification","Association Rules","Prediction"])
    if page=="Overview":
        st.markdown("<span style='color:#42d6c4;letter-spacing:.15em;font-size:.8rem'>INTERACTIVE ANALYTICS WORKSPACE</span>",unsafe_allow_html=True)
        st.title("OULAD Insight Lab")
        st.write("Explore student outcomes, engagement segments, activity patterns and assessment score predictions.")
        required=["studentInfo.csv","studentVle.csv","vle.csv","assessments.csv","studentAssessment.csv"]
        p=Path(folder) if folder else None
        found=sum((p/f).exists() for f in required) if p else 0
        c=st.columns(4)
        c[0].metric("Required files",f"{found}/{len(required)}")
        c[1].metric("Analytics modules","4")
        c[2].metric("Clustering methods","3")
        c[3].metric("Local execution","Ready" if found==5 else "Check files")
        st.markdown("---")
        left,right=st.columns([1.3,1])
        with left:
            st.subheader("Project workspace")
            st.write("A local website for interactive parameter selection, model comparison, visual analysis and CSV export.")
            st.write("**Clustering:** K-means, Hierarchical, DBSCAN")
            st.write("**Classification:** Dummy, Decision Tree, Random Forest, Extra Trees, Logistic Regression")
            st.write("**Association rules:** user-controlled support, confidence and lift")
            st.write("**Prediction:** baseline, Linear/Ridge Regression, Decision Tree, Random Forest, Extra Trees")
        with right:
            st.subheader("Dataset readiness")
            for f in required:st.write(("🟢 " if p and (p/f).exists() else "🟠 ")+f)
            if found<5:
                st.warning("Choose a valid local dataset folder or upload all five required CSV files in the sidebar.")
        st.caption("Metrics are computed at runtime and can differ from previous experiments because preprocessing and splits may differ.")
    elif not folder:
        st.title("Dataset required")
        st.info("Use the sidebar to select a dataset folder or upload all five required OULAD CSV files.")
    elif page=="Clustering":clustering(folder)
    elif page=="Classification":classifier(folder)
    elif page=="Association Rules":rules(folder)
    else:prediction(folder)

if __name__=="__main__":main()
