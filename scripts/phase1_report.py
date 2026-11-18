"""Generate first-evaluation evidence from actual saved run outputs."""
import json, html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from pathsnap.data import load_trip
from pathsnap.reference import load_reference
from pathsnap.features import FEATURE_NAMES


def plot_results(out, table, inventory):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=table.model.str.replace('_','\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['cv_macro_f1','top1_accuracy','macro_f1','log_loss'],
                            ['Group-CV candidate macro F1','Student-test observation Top-1','Student-test candidate macro F1','Student-test binary log loss']):
        ax.bar(labels,table[key],color='#087e8b');ax.set_title(title);ax.tick_params(axis='x',labelsize=8)
        for i,value in enumerate(table[key]):ax.text(i,value,f'{value:.3f}',ha='center',va='bottom',fontsize=8)
        ax.set_ylim(0,max(table[key])*1.18)
    fig.savefig(out/'model_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
    axes[0].hist(inventory.observations,bins=25,color='#087e8b');axes[0].set_title('GPS points per trip');axes[0].set_xlabel('Both supplied partitions')
    axes[1].hist(inventory.reference_distance_p95,bins=25,color='#db9b24');axes[1].set_title('95th-percentile reference distance');axes[1].set_xlabel('metres; derived alignment')
    axes[2].scatter(inventory.observations,inventory.candidate_miss_rate*100,c='#5365b1');axes[2].set_title('Candidate misses by trip');axes[2].set_xlabel('GPS observations');axes[2].set_ylabel('Miss rate (%)')
    fig.savefig(out/'eda.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for ax,(_,r) in zip(axes.flat,table.iterrows()):
        cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],int)
        ax.imshow(cm,cmap='Blues');ax.set_title(r.model,fontsize=10)
        for i in range(2):
            for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        ax.set_xticks([0,1],['negative','positive']);ax.set_yticks([0,1],['negative','positive'])
        ax.set_xlabel('Predicted');ax.set_ylabel('Reference proxy')
    fig.savefig(out/'confusion_matrices.png',dpi=160);plt.close(fig)


def make_map(out,model):
    import folium
    trip_id='00000007'
    folder=Path('data/Student folder/student_test')/trip_id
    trip=load_trip(folder);reference=load_reference(folder,len(trip.arcs))
    with np.load(f'results/baseline/predictions/{model}_{trip_id}.npz') as z: predicted=z['predicted']
    center=trip.track[:,:2].mean(axis=0)[::-1].tolist()
    m=folium.Map(location=center,zoom_start=13,tiles='OpenStreetMap')
    road=folium.FeatureGroup('Road network',show=True)
    # GeoJSON MultiLineString is compact and avoids thousands of map widgets.
    roadgeo={'type':'Feature','geometry':{'type':'MultiLineString','coordinates':trip.nodes[trip.arcs].tolist()},'properties':{}}
    folium.GeoJson(roadgeo,style_function=lambda _:dict(color='#758294',weight=1,opacity=.35)).add_to(road)
    road.add_to(m)
    for name,ids,color,weight in [('Reference route (development only)',reference,'#d99410',6),('Classifier prediction',np.unique(predicted),'#db4264',3)]:
        layer=folium.FeatureGroup(name)
        geo={'type':'Feature','geometry':{'type':'MultiLineString','coordinates':trip.nodes[trip.arcs[ids]].tolist()},'properties':{}}
        folium.GeoJson(geo,style_function=lambda _,c=color,w=weight:dict(color=c,weight=w,opacity=.8)).add_to(layer);layer.add_to(m)
    layer=folium.FeatureGroup('GPS observations')
    for p in trip.track[::max(1,len(trip.track)//120)]:
        folium.CircleMarker(location=p[:2][::-1],radius=2,weight=1,color='#1879c9',fill=True,fill_opacity=.65).add_to(layer)
    layer.add_to(m);folium.LayerControl().add_to(m)
    m.fit_bounds([trip.track[:,:2].min(axis=0)[::-1].tolist(),trip.track[:,:2].max(axis=0)[::-1].tolist()])
    legend='<div style="position:fixed;top:20px;left:70px;z-index:9999;background:white;padding:14px;font:14px Arial;border-radius:8px"><b>PathSnap - student-test trip 00000007</b><br><span style="color:#1879c9">Blue: GPS</span><br><span style="color:#d99410">Gold: reference (development only)</span><br><span style="color:#db4264">Pink: classifier</span><br>Point labels are derived from the ordered route.</div>'
    m.get_root().html.add_child(folium.Element(legend));m.save(str(out/'development_map.html'))
    fig,ax=plt.subplots(figsize=(9,5.7),layout='constrained')
    ax.add_collection(LineCollection(trip.nodes[trip.arcs],colors='#c7d1d8',linewidths=.35))
    ax.add_collection(LineCollection(trip.nodes[trip.arcs[reference]],colors='#d99410',linewidths=3,label='Reference route'))
    ax.add_collection(LineCollection(trip.nodes[trip.arcs[np.unique(predicted)]],colors='#db4264',linewidths=1.4,label='Classifier prediction'))
    ax.scatter(trip.track[:,0],trip.track[:,1],s=2,c='#1879c9',label='GPS observations')
    lo,hi=trip.track[:,:2].min(axis=0),trip.track[:,:2].max(axis=0);pad=(hi-lo)*.06
    ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect(1/np.cos(np.deg2rad(center[0])))
    ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.set_title('Student-test trip 00000007 - geographic overlay');ax.legend(fontsize=8)
    fig.savefig(out/'geographic_overlay.png',dpi=160);plt.close(fig)


def main():
    out=Path('output/phase1');out.mkdir(parents=True,exist_ok=True)
    table=pd.read_csv('results/baseline/model_comparison.csv')
    inventory=pd.read_csv('results/dataset_inventory.csv',dtype={'trip_id':str})
    metadata=json.loads(Path('results/baseline/selection.json').read_text())
    config=json.loads(Path('results/baseline/run_config.json').read_text())
    selected=metadata['model_name'];best=table.loc[table.model==selected].iloc[0]
    plot_results(out,table,inventory);make_map(out,selected)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='SmallTable',fontSize=7.4,leading=9.8))
    styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=13
    story=[]
    def para(text,style='BodyText'):story.append(Paragraph(text,styles[style]));story.append(Spacer(1,7))
    def heading(text):para(text,'Heading2')
    def pic(file,width=470):
        from PIL import Image as PILImage
        w,h=PILImage.open(out/file).size;story.append(Image(str(out/file),width=width,height=width*h/w));story.append(Spacer(1,9))
    def grid(rows,widths):
        cells=[[Paragraph(html.escape(str(c)),styles['SmallTable']) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0eef1')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#ccd4dd')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);story.append(Spacer(1,10))
    para('PathSnap | First evaluation','Title')
    para('Version 0 evidence package - 18 November 2026','Heading2')
    para('Scope: data preprocessing and feature engineering (5 marks); model selection and training (5 marks). Prepared for the team\'s 11:00 AM target; the teacher\'s first evaluation is scheduled for 11:30 AM IST.')
    heading('Measured result')
    para(f'Selected baseline: <b>{selected}</b>, chosen by mean trip-group cross-validation observation Top-1 accuracy. Student-test road-selection accuracy: <b>{best.top1_accuracy:.2%}</b>. Candidate binary accuracy: {best.accuracy:.2%}; precision: {best.precision:.2%}; recall: {best.recall:.2%}; specificity: {best.specificity:.2%}; macro F1: {best.macro_f1:.4f}.')
    para('<b>Metric limitation:</b> .route contains an ordered arc sequence, not timestamped GPS-point labels. Observation labels are derived with monotone distance alignment. These are proxy-label development metrics, not final evaluator scores. Binary accuracy includes many negative candidates and is not the road-selection accuracy.')
    heading('Reproducible evidence')
    grid([['Evidence','Location'],['Data audit + source checksums','results/dataset_inventory.csv; results/data_manifest.json'],['All six model results + fold trip IDs','results/baseline/model_comparison.csv; cross_validation.json'],['Every trained model + selected artifact','results/baseline/models/; artifacts/model.joblib'],['Feature definitions + label derivation','pathsnap/features.py; pathsnap/reference.py'],['Interactive development map','output/phase1/development_map.html']], [155,315])
    para('AI assistance was used for implementation, execution and this draft. The saved metrics come from actual runs. Each teammate must review the explanations, reproduce their assigned part, and follow the event\'s disclosure requirements. No personal contribution claims are fabricated.')
    story.append(PageBreak())
    heading('1. Dataset understanding and preprocessing')
    sums=inventory.groupby('partition').agg(trips=('trip_id','count'),gps=('observations','sum'),arcs=('arcs','sum'))
    grid([['Partition','Trips','GPS points','Road arcs (trip-local)']]+[[idx,int(r.trips),int(r.gps),int(r.arcs)] for idx,r in sums.iterrows()],[130,70,110,160])
    para('The evaluator-testing partition is not supplied. No evaluator data were inspected. Node IDs and directed arc IDs are zero-based, trip-local row indices. .track columns are longitude, latitude and seconds; .nodes stores longitude/latitude; .arcs stores directed start/end nodes; .route stores an ordered sequence of arc IDs.')
    para('The parser validates dimensions, finite values, geographic bounds, node references and nondecreasing timestamps. It preserves point order and does not silently remove samples. Local equirectangular projection converts degrees to metres using each trip\'s mean latitude. This approximates local geometry; large geographic extents would warrant a geodesic or appropriate projected CRS.')
    para(f'Finite input audit: {int(inventory.missing_values.sum())} missing/non-finite values in accepted trips. Duplicate timestamp steps: {int(inventory.duplicate_timestamps.sum())}. Steps above 10 seconds: {int(inventory.gaps_over_10s.sum())}; largest gap: {inventory.max_gap_s.max():.0f} seconds. Disconnected road components and isolated nodes are recorded per trip.')
    pic('eda.png')
    heading('Data-quality limitations')
    para('Training trips 00000033 and 00000054 have large disagreement between GPS and the supplied reference alignment: about 66% and 68% of derived reference arcs are absent from the eight nearest candidates. All trips and misses remain in the reported audit. We do not inject the correct arc into candidates or remove test failures. This reveals noisy supervision and a candidate-recall ceiling.')
    story.append(PageBreak())
    heading('2. Candidate generation and features')
    para('Each GPS point receives the eight nearest directed line segments. A KD-tree over segment samples retrieves an initial pool; a radius expansion using the segment-sampling distance bound guarantees the nearest segments under the local projection. Reverse directions remain distinct candidates. Ties are deterministic by arc ID.')
    grid([['Feature group','Features and purpose'],['Distance','Log and clipped perpendicular distance, nearest-candidate gap and rank: local geometric plausibility.'],['Heading and motion','5- and 15-observation-window heading cosine, movement, speed and directed forward speed: distinguish reverse arcs and parallel roads.'],['Segment position','Length, projection fraction and distances to endpoints: describe progress near intersections.'],['Temporal context','Distance from observations three steps before/after to this segment; timestamp gap: nearby trajectory consistency.'],['Network structure','Start/end outdegree: describe intersections without using trip IDs or absolute arc IDs.']], [120,350])
    para('18 numerical features are shared by every model. The candidate input pipeline has no .route access. Route labels are loaded only in training/scoring. A monotone dynamic program associates observations with nondecreasing positions in the supplied route; labels are never appended to prediction features. Features use both past and future GPS context, so version 0 is an offline whole-trip matcher.')
    para(f'Training uses {config["observations"]:,} sampled observations ({config["candidate_rows"]:,} candidate rows) from all 80 training trips, capped at {config["max_obs_per_trip"]:,} observations per trip with seed {config["seed"]}. This limits long-trip dominance and first-phase runtime. Student-test scoring uses every one of its 19,160 observations. Three-fold GroupKFold keeps entire trips in a single fold. Scalers are fitted inside each fold. No random point-level validation or validation-based early stopping is used.')
    heading('Metric definitions')
    para('Positive class: candidate arc equals the derived reference arc. Candidate metrics use probability threshold 0.5: precision = TP/(TP+FP); recall = TP/(TP+FN); specificity = TN/(TN+FP); accuracy = (TP+TN)/all candidates. Candidate macro F1 averages F1 for the two classes. Observation Top-1 selects the highest-scoring candidate for each GPS point and counts candidate misses as failures.')
    para('Strict route continuity means consecutive predictions are identical arcs or share a directed endpoint. It penalizes legitimate multi-edge jumps and is a diagnostic, not a full route-quality measure. The comparison files also retain edit similarity and route-arc overlap.')
    story.append(PageBreak())
    heading('3. Model comparison')
    grid([['Model','CV Top-1','CV macro F1','Test Top-1','Test macro F1','Log loss','Fit s']]+[[r.model,f'{r.cv_top1_accuracy:.3f}',f'{r.cv_macro_f1:.3f}',f'{r.top1_accuracy:.3f}',f'{r.macro_f1:.3f}',f'{r.log_loss:.3f}',f'{r.train_seconds:.1f}'] for _,r in table.iterrows()],[110,60,60,60,60,60,60])
    pic('model_comparison.png')
    story.append(PageBreak())
    heading('4. Model-specific interpretation')
    descriptions={
     'logistic':('Linear probabilistic','C=1; max_iter=400; standardized inputs. Optimizes regularized logistic loss.','Fast and interpretable, but one linear boundary cannot capture all distance-heading interactions.'),
     'gaussian_nb':('Probabilistic generative','var_smoothing=1e-7; standardized inputs. Fits class-conditional Gaussian likelihoods.','Fast baseline; strong conditional-independence and Gaussian assumptions conflict with correlated geometric features.'),
     'id3':('ID3 decision tree','Six quantile bins per feature fitted on training data; max_depth=8; min_samples_split=100; Shannon information gain; Laplace alpha=1.','Multiway ID3 splits capture nonlinear rules. Discretization loses some precision; limited depth and minimum node support regularize the tree.'),
     'random_forest':('Bagged tree ensemble','80 trees; max_depth=18; min_samples_leaf=12. Gini splits and probability averaging.','Reduces individual-tree variance; training and artifact size are larger than linear models.'),
     'knn':('K-nearest neighbours','15 neighbours; distance weighting; standardized features; exact KD-tree search. No parametric loss is minimized: the training set is retained for neighbourhood inference.','Local geometric similarity is easy to explain. Correlated features, noisy labels and expensive inference are limitations.'),
     'gradient_boosting':('Gradient Boosting','100 trees; learning_rate=0.08; max_depth=2; min_samples_leaf=30; subsample=0.7; sqrt feature subsampling. Binary log loss.','Sequential weak-tree corrections learn nonlinear rules; shallow trees can underfit and many iterations can fit noisy labels.')}
    for model_index,(_,r) in enumerate(table.iterrows()):
        if model_index == 3:
            story.append(PageBreak())
            heading('4. Model-specific interpretation (continued)')
        family,training,reason=descriptions[r.model]
        heading(r.model+' - '+family)
        para(training+' '+reason)
        para(f'Measured training Top-1 {r.train_top1_accuracy:.2%}; CV Top-1 {r.cv_top1_accuracy:.2%} +/- {r.cv_top1_std:.2%} across folds; test Top-1 {r.top1_accuracy:.2%}; test macro F1 {r.macro_f1:.4f}; log loss {r.log_loss:.4f}; classifier inference {r.inference_seconds:.3f}s for all test candidates. Training-minus-CV gap: {(r.train_top1_accuracy-r.cv_top1_accuracy)*100:.2f} percentage points. This gap is evidence of generalization loss; it does not by itself distinguish label noise from overfitting.')
    story.append(PageBreak())
    heading('5. Diagnostics and selection')
    para(f'{selected} has the highest mean group-CV Top-1 score among the six runs. This is the primary metric because the task selects one road per GPS point; binary accuracy can be inflated by negatives. CV standard deviation, candidate log loss and inference time are supporting diagnostics. Student testing is a development comparison set, not an unseen final evaluation.')
    pic('confusion_matrices.png')
    grid([['Selected model metric','Measured value']]+[[k,f'{float(best[k]):.6f}'] for k in ['accuracy','precision','recall','specificity','f1','macro_f1','top1_accuracy','candidate_miss_rate','strict_adjacent_continuity','log_loss']],[240,230])
    story.append(PageBreak())
    heading('6. Geographical evidence')
    para('Student-test trip 00000007 is visualized for development. The accompanying Folium HTML includes road geometry, GPS, reference route and classifier predictions with layer controls. Reference geometry belongs only in offline development maps, never in the evaluator UI response.')
    pic('geographic_overlay.png')
    if (out/'folium_screenshot.jpg').exists():pic('folium_screenshot.jpg')
    else:para('The figure above is a static geographic export of the same data layers. The interactive Folium map is included separately. A browser screenshot is added when captured; this static plot is not represented as a Folium screenshot.')
    story.append(PageBreak())
    heading('7. First-evaluation demo and next milestones')
    para('1. Show results/dataset_inventory.csv and explain the four raw file types. 2. Open pathsnap/features.py and explain candidate generation, metre projection and heading features. 3. Show model_comparison.csv and cross_validation.json. 4. Explain why Top-1 and candidate accuracy differ. 5. Load artifacts/model.joblib and reproduce a prediction; show the Folium map and this report.')
    grid([['Time (IST)','Target'],['Before 11:00 AM','First-phase evidence, trained baseline, source, model comparisons and team walkthrough.'],['11:30 AM','Teacher\'s first evaluation: preprocessing/features and model selection/training.'],['After first-phase package','Bayesian tuning, HMM/Viterbi comparison, FastAPI and separate UI integration.'],['Before 3:00 PM','Two-laptop network rehearsal; final report; all file submission through the teacher\'s Google Form.'],['4:30 PM / 5:00 PM','Form closes / hackathon ends. Treat 3:00 PM as the submission target.']], [115,355])
    para('Version 0 first-phase evidence does not claim that Bayesian tuning, two-laptop deployment, reserved evaluator testing, GitHub publishing or Google Form submission have already occurred.')
    heading('Four-teammate ownership')
    grid([['Owner','Scope and handoff'],['Teammate 1 - data','Parsing, EDA, candidate generation, feature definitions, data manifest.'],['Teammate 2 - training','Six models, grouped CV, measured metrics, selected model, Bayesian tuning.'],['Teammate 3 - model/API','HMM/Viterbi, inference artifact contract, server-side scoring, FastAPI on Laptop 1.'],['Teammate 4 - UI/report','API client and evaluator UI on Laptop 2, Folium maps, consolidated report and demo evidence.']], [140,330])
    heading('References')
    para('Syllabus compliance: Logistic Regression, ID3 Decision Trees and KNN appear in Unit 1. Random Forest, Gradient Boost and Naive Bayes appear under Boosting/Bayesian Learning/HMM. HMM and Bayesian Optimization are explicitly listed for the later phase. The comparison uses only these syllabus methods. An earlier prototype was superseded after the syllabus was supplied; its scores are not mixed into this comparison.')
    para('Project requirements: the five supplied PathSnap PDFs. Technical references: scikit-learn GroupKFold documentation (scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html); Optuna TPESampler documentation (optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html); Newson and Krumm (2009), Hidden Markov Map Matching Through Noise and Sparseness, Microsoft Research. No external model weights or additional training datasets were used.')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536273'));canvas.drawString(42,25,'PathSnap | Version 0 | First evaluation | Measured development results');canvas.drawRightString(553,25,str(doc.page))
    SimpleDocTemplate(str(out/'PathSnap_First_Evaluation.pdf'),pagesize=(595,842),rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print('Report generated:',out/'PathSnap_First_Evaluation.pdf')


if __name__=='__main__':main()
