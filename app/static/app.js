'use strict';
let overview={},schema={fields:[],samples:[],thresholds:{}},current=null;
const $=id=>document.getElementById(id);
const fmt=n=>new Intl.NumberFormat('en-US').format(n??0);
const pct=(n,d=2)=>n==null||isNaN(n)?'—':(n*100).toFixed(d)+'%';
const num=(n,d=3)=>n==null||isNaN(n)?'—':Number(n).toFixed(d);
const esc=s=>String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const api=async(path,options={})=>{const r=await fetch(path,{...options,headers:{...(options.headers||{}),'X-Requested-With':'fraudguard-ui'}});const d=await r.json().catch(()=>({}));if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:`HTTP ${r.status}`);return d};
const titles={dashboard:'نظرة عامة',screen:'فحص عملية',batch:'فحص دفعة',history:'سجل الفحوصات',models:'التقييم والتجارب',about:'عن النظام'};
const decAr={approve:'قبول',review:'مراجعة يدوية',block:'حظر'};
const expAr={none:'بدون معالجة',class_weight:'أوزان الفئات','undersample_1:10':'تقليل العينات 1:10','smote_1:10':'SMOTE 1:10',main:'النموذج الرئيسي',without_behavioural:'بدون خصائص سلوك الحسابات',with_simulator_artifacts:'مع ثغرة المحاكي (خطأ الأرصدة)',without_balance_fields:'بدون حقول الأرصدة',without_amount_time:'بدون المبلغ والوقت'};
const rawCols={paysim:'step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest [, isFraud]',ulb:'Time, V1…V28, Amount [, Class]',samld:'Time, Date, Sender_account, Receiver_account, Amount, Payment_currency, Received_currency, Sender_bank_location, Receiver_bank_location, Payment_type [, Is_laundering]'};
const figures=[['fig_pr_roc.png','منحنيات PR و ROC'],['fig_shap.png','أهم الخصائص (SHAP)'],['fig_drift.png','الثبات الزمني'],['fig_confusion.png','مصفوفة الالتباس']];

function goto(v){document.querySelectorAll('.view').forEach(e=>e.classList.toggle('current',e.id==='view-'+v));document.querySelectorAll('.nav-link').forEach(e=>e.classList.toggle('active',e.dataset.view===v));$('breadcrumb').textContent=titles[v]||v;$('sidebar').classList.remove('open');if(v==='history')loadHistory();window.scrollTo({top:0,behavior:'smooth'})}
function toast(m){$('toast').textContent=m;$('toast').classList.remove('hidden');setTimeout(()=>$('toast').classList.add('hidden'),3500)}
const D=()=>overview.datasets?.[current]||{};

function drawDashboard(){
  const ds=overview.datasets||{};
  $('dsCards').innerHTML=Object.entries(ds).map(([k,d])=>{const f=d.final_test||{};return `<button class="ds-card ${k===current?'active':''}" data-ds="${esc(k)}"><span class="ds-crime">${esc(d.crime_ar)}</span><strong>${esc(d.title_ar)}</strong><div class="ds-metrics"><div><b>${pct(f.pr_auc,1)}</b><small>PR-AUC</small></div><div><b>${pct(f.recall_at_1fpr,1)}</b><small>كشف عند 1% إنذار</small></div><div><b>${fmt(d.rows)}</b><small>عملية · ${pct(d.prevalence,2)} جريمة</small></div></div><small class="english">${esc(d.selected)}</small></button>`}).join('');
  const c=overview.cross_dataset||{},ranks=Object.entries(c.average_ranks||{});
  $('friedmanBadge').textContent=c.friedman_p!=null?`Friedman p = ${c.friedman_p} · CD = ${c.nemenyi_cd}`:'—';
  $('rankRows').innerHTML=ranks.map(([a,r],i)=>`<div class="progress-row"><div class="progress-info"><span class="english">${esc(a)}</span><span>${num(r,2)}</span></div><div class="track"><div class="bar ${i===0?'primary':'light'}" style="width:${(7-r)/6*100}%"></div></div></div>`).join('')||'<p class="body-p">تظهر بعد تدريب المجموعات الثلاث.</p>';
  const dc=overview.decisions||{};
  $('screenStats').innerHTML=[['الإجمالي',overview.screenings],['قبول',dc.approve],['مراجعة',dc.review],['حظر',dc.block]].map(([a,b])=>`<div class="stat"><div class="stat-top"><span>${a}</span></div><strong>${fmt(b||0)}</strong></div>`).join('');
  $('aboutRows').innerHTML=Object.entries(ds).map(([k,d])=>`<tr><td>${esc(d.title_ar)}</td><td>${esc(d.crime_ar)}</td><td>${fmt(d.rows)}</td><td>${pct(d.prevalence,2)}</td><td><a class="english" href="${esc(d.source)}" target="_blank" rel="noopener">Kaggle</a></td></tr>`).join('');
}

function drawModels(){
  const d=D(),f=d.final_test||{},sel=d.selected||'';
  $('selectedBadge').textContent=sel;
  const comp=Object.entries(d.test_comparison||{}),best=comp[0]?.[0];
  $('metricsRows').innerHTML=comp.map(([a,x])=>`<tr><td class="english title-cell">${esc(a)}${a===best?' ★':''}</td><td><b>${num(x.pr_auc)}</b> <small class="muted">± ${num(x.pr_auc_std)}</small></td><td>${num(x.roc_auc)}</td><td>${pct(x.recall_at_1fpr,1)}</td><td>${pct(x.recall_at_01fpr,1)}</td></tr>`).join('');
  $('significance').textContent=f.diff_p_value!=null?`مقابل ${d.runner_up}: p = ${f.diff_p_value}`:'—';
  const opName={review_1fpr:'مراجعة (1% FPR)',block_01fpr:'حظر (0.1% FPR)'};
  $('opRows').innerHTML=Object.entries(d.operating_points||{}).map(([k,x])=>`<tr><td>${opName[k]||k}</td><td>${pct(x.recall,1)}</td><td>${pct(x.precision,1)}</td><td>${pct(x.fpr,3)}</td><td>${fmt(x.tp)}</td><td>${fmt(x.fp)}</td><td>${fmt(x.fn)}</td><td>${num(x.alerts_per_10k,1)}</td></tr>`).join('');
  const ex=[...Object.entries(d.imbalance||{}).map(([k,v])=>['imb',k,v]),...Object.entries(d.ablation||{}).filter(([k])=>k!=='main').map(([k,v])=>['abl',k,v])];
  $('experimentRows').innerHTML=ex.map(([g,k,v])=>`<div class="progress-row"><div class="progress-info"><span>${g==='abl'?'استئصال: ':''}${esc(expAr[k]||k)}</span><span>${num(v)}</span></div><div class="track"><div class="bar ${g==='abl'?'':'light'}" style="width:${v*100}%"></div></div></div>`).join('');
  $('periodRows').innerHTML=(d.periods||[]).map(p=>`<tr><td>${p.period}</td><td>${pct(p.prevalence,2)}</td><td>${num(p.pr_auc)}</td><td>${pct(p.fpr_at_review,2)}</td></tr>`).join('');
  const dr=(d.drift_psi||[]).slice(0,3);$('driftText').innerHTML=dr.length?`أعلى انحراف (PSI): ${dr.map(x=>`${esc(x.label_ar)} <b>${num(x.psi,2)}</b>`).join('، ')}`:'';
  const v=encodeURIComponent(overview.trained_at||'');
  $('figures').innerHTML=[...figures.map(([f,c])=>[`${current}/${f}`,c]),['fig_summary_heatmap.png','ملخص PR-AUC عبر المجموعات']].map(([f,c])=>`<figure data-fig="/reports/${f}?v=${v}"><img src="/reports/${f}?v=${v}" alt="${esc(c)}" loading="lazy" onerror="this.closest('figure').remove()"><figcaption>${esc(c)}</figcaption></figure>`).join('');
}

async function loadSchema(){
  schema=await api('/api/schema/'+current);
  $('screenModel').textContent=schema.model;$('screenSubtitle').textContent=schema.title_ar+' — اختر مثالاً من بيانات الاختبار أو أدخل القيم.';
  $('batchColumns').textContent=rawCols[current]||'—';
  $('formGrid').innerHTML=schema.fields.map(f=>{const id='f_'+f.name;let input;
    if(f.type==='category')input=`<select id="${id}">${f.options.map(o=>`<option ${o===f.default?'selected':''}>${esc(o)}</option>`).join('')}</select>`;
    else if(f.type==='binary')input=`<select id="${id}"><option value="0" ${f.default==0?'selected':''}>0</option><option value="1" ${f.default==1?'selected':''}>1</option></select>`;
    else input=`<input id="${id}" type="number" step="any" value="${f.default??''}">`;
    return `<label><b>${esc(f.label_ar)}</b><span class="english">${esc(f.name)}</span>${input}</label>`}).join('');
  $('sampleButtons').innerHTML=schema.samples.map((s,i)=>`<button class="chip" data-sample="${i}">${esc(s.label)}</button>`).join('');
  $('resultEmpty').classList.remove('hidden');$('resultContent').classList.add('hidden');$('resultStatus').textContent='بانتظار الفحص';
}
function fillSample(i){const s=schema.samples[i];if(!s)return;for(const [k,v] of Object.entries(s.features)){const el=$('f_'+k);if(el)el.value=v==null?'':String(v)}toast('تم تحميل: '+s.label)}

async function runScore(){
  $('formError').textContent='';const features={};
  for(const f of schema.fields){const v=$('f_'+f.name).value;features[f.name]=f.type==='category'?v:(v===''?null:Number(v))}
  const btn=$('scoreBtn');btn.disabled=true;btn.textContent='جاري الفحص...';
  try{
    const r=await api('/api/score/'+current,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({features})});
    $('resultEmpty').classList.add('hidden');$('resultContent').classList.remove('hidden');
    $('decisionCard').className='decision-card '+r.decision;$('decisionLabel').textContent=decAr[r.decision];
    $('resultStatus').textContent=decAr[r.decision];$('probValue').textContent=pct(r.probability,2);
    const pos=s=>Math.min(100,Math.max(0,(Math.log10(Math.max(s,1e-6))+6)/6*100));
    $('riskFill').style.width=pos(r.score)+'%';$('tickReview').style.left=pos(r.thresholds.review)+'%';$('tickBlock').style.left=pos(r.thresholds.block)+'%';
    const mx=Math.max(...r.reasons.map(x=>Math.abs(x.contribution)),1e-6);
    $('reasons').innerHTML=r.reasons.map(x=>`<div class="reason"><div>${esc(x.label_ar)}<small>${esc(x.feature)}${x.value!=null?' = '+esc(typeof x.value==='number'?+x.value.toFixed(4):x.value):''}</small></div><div class="reason-bar"><div class="${x.contribution>0?'up':'down'}" style="width:${Math.abs(x.contribution)/mx*50}%"></div></div></div>`).join('');
    $('resultMeta').textContent=`#${r.id} · ${r.model}`+(r.defaulted_fields.length?` · حقول بقيمة افتراضية: ${r.defaulted_fields.length}`:'');
    overview.screenings=(overview.screenings||0)+1;
  }catch(e){$('formError').textContent='تعذّر الفحص: '+e.message}
  finally{btn.disabled=false;btn.textContent='◇ تشغيل الفحص'}
}

async function runBatch(){
  $('batchError').textContent='';const file=$('batchFile').files[0];if(!file){$('batchError').textContent='اختر ملف CSV أولاً.';return}
  const fd=new FormData();fd.append('file',file);const btn=$('batchBtn');btn.disabled=true;btn.textContent='جاري الفحص...';
  try{
    const r=await api('/api/score-batch/'+current,{method:'POST',body:fd});$('batchResult').classList.remove('hidden');const d=r.decisions;
    $('batchStats').innerHTML=[['عمليات مفحوصة',fmt(r.rows_scored)+' / '+fmt(r.rows_in_file)],['قبول',fmt(d.approve)],['مراجعة',fmt(d.review)],['حظر',fmt(d.block)]].map(([a,b])=>`<div class="stat"><div class="stat-top"><span>${a}</span></div><strong>${b}</strong></div>`).join('');
    const e=r.evaluation;$('batchEval').innerHTML=e?`<div class="notice"><span class="notice-icon">✓</span><div><strong>تقييم مباشر</strong><p>اكتُشف ${fmt(e.caught)} من ${fmt(e.positives)} حالة (Recall ${pct(e.recall,1)}) · Precision ${pct(e.precision,1)} · إنذارات كاذبة ${fmt(e.false_alarms)} (FPR ${pct(e.fpr,3)})</p></div></div>`:'';
    $('batchRows').innerHTML=r.top_risk.map(x=>`<tr><td>${x.row}</td><td>${pct(x.probability,2)}</td><td><span class="dec-badge ${x.decision}">${decAr[x.decision]}</span></td><td>${x.actual==null?'—':x.actual?'جريمة':'سليمة'}</td></tr>`).join('');
  }catch(e){$('batchError').textContent='تعذّر الفحص: '+e.message}
  finally{btn.disabled=false;btn.textContent='تشغيل فحص الدفعة'}
}

async function loadHistory(){try{const h=await api('/api/history?limit=100');$('historyCount').textContent=`إجمالي ${fmt(h.total)} فحص`;$('historyRows').innerHTML=h.items.length?h.items.map(r=>`<tr><td>${esc(new Date(r.created_at).toLocaleString('ar-SA'))}</td><td class="english">${esc(r.dataset)}</td><td><span class="dec-badge ${r.decision}">${decAr[r.decision]}</span></td><td>${pct(r.probability,2)}</td><td>${r.top_reasons.map(x=>esc(x.label_ar)).join('، ')}</td></tr>`).join(''):'<tr><td colspan="5">لا توجد فحوصات بعد.</td></tr>'}catch(e){toast(e.message)}}
async function clearHistory(){if(!confirm('حذف كل الفحوصات المحفوظة؟ لا يمكن التراجع.'))return;try{await api('/api/history',{method:'DELETE'});overview.screenings=0;overview.decisions={};drawDashboard();loadHistory();toast('تم حذف السجل')}catch(e){toast(e.message)}}

async function selectDataset(k){current=k;$('datasetSelect').value=k;drawDashboard();drawModels();try{await loadSchema()}catch(e){toast(e.message)}}
async function init(){
  try{overview=await api('/api/overview');const keys=Object.keys(overview.datasets||{});if(!keys.length)throw Error('لا توجد نماذج مدربة. شغّل python train.py');
    $('datasetSelect').innerHTML=keys.map(k=>`<option value="${esc(k)}">${esc(overview.datasets[k].title_ar)}</option>`).join('');
    await selectDataset(keys[0]);$('sideModelStatus').textContent=`${keys.length} نماذج جاهزة`}
  catch(e){$('sideModelStatus').textContent='تعذّر تحميل النماذج';toast(e.message)}
}
document.querySelectorAll('[data-view]').forEach(e=>e.addEventListener('click',()=>goto(e.dataset.view)));
document.querySelectorAll('[data-go]').forEach(e=>e.addEventListener('click',()=>goto(e.dataset.go)));
$('datasetSelect').addEventListener('change',e=>selectDataset(e.target.value));
$('dsCards').addEventListener('click',e=>{const b=e.target.closest('[data-ds]');if(b)selectDataset(b.dataset.ds)});
$('sampleButtons').addEventListener('click',e=>{const b=e.target.closest('[data-sample]');if(b)fillSample(Number(b.dataset.sample))});
$('scoreBtn').addEventListener('click',runScore);$('batchBtn').addEventListener('click',runBatch);
$('figures').addEventListener('click',e=>{const f=e.target.closest('[data-fig]');if(f){$('figImg').src=f.dataset.fig;$('figOverlay').classList.remove('hidden')}});
$('figOverlay').addEventListener('click',()=>$('figOverlay').classList.add('hidden'));
document.addEventListener('keydown',e=>{if(e.key==='Escape')$('figOverlay').classList.add('hidden')});
$('clearHistory').addEventListener('click',clearHistory);$('menuToggle').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
$('clock').textContent=new Date().toLocaleDateString('ar-SA',{month:'long',day:'numeric',year:'numeric'});
init();
