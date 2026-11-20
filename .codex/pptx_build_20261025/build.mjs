import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const ROOT='C:/Users/siomi/OneDrive/デスクトップ/Kenkyuu';
const TMP=ROOT+'/.codex/pptx_build_20261025';
const SKILL='C:/Users/siomi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const {applyPresentationChartFont}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const FONT='Meiryo';
const C={navy:'#175D76',blue:'#355EAD',green:'#16816B',orange:'#D58A22',purple:'#9962A5',ink:'#172D38',gray:'#5C6B73',light:'#DDE5E8'};
const src3=TMP+'/第3回　カメラ班ミーティング/';
const src4=TMP+'/第4回　カメラ班ミーティング/';
const report=ROOT+'/output/paf_1194m_no_surface_v1/REPORT.md';
const core=ROOT+'/output/paf_core_ablation_v2/';
const diagnosis=ROOT+'/output/paf_training_diagnosis_v1/REPORT.md';
let count=0;

// 全ページで文字・余白・出典の位置を揃える。
function text(slide,content,x,y,w,h,size=28,color=C.ink,bold=false){
  const s=slide.shapes.add({geometry:'textbox',name:'text-'+(++count),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  s.text=content; s.text.style={fontFamily:FONT,fontSize:size,color,bold}; s.text.insets=0;
  return s;
}
function rule(slide,x,y,w,color=C.navy,h=3){slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:color,line:{fill:'none',width:0}});}
function page(p,title,section,notes=''){
  const s=p.slides.add();s.background.fill='#FFFFFF';
  text(s,section,50,23,1100,28,18,C.navy,true);
  text(s,title,50,66,1180,67,39,C.ink,true);rule(s,50,143,1180);
  rule(s,50,680,1180,C.light,1);
  text(s,'PAF内周検出　進捗報告',50,690,800,23,14,C.gray);
  text(s,String(p.slides.items.length),1180,687,50,28,18,C.gray);
  s.speakerNotes.text=notes;return s;
}
function bottom(s,t){text(s,t,55,612,1170,60,25,C.navy,true);}
function note(s,t){text(s,t,55,573,1170,32,16,C.gray);}
async function img(s,file,x,y,w,h,alt,crop){
  const bytes=await fs.readFile(file);
  return s.images.add({blob:bytes,contentType:file.endsWith('.jpg')?'image/jpeg':'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h},...(crop?{crop}: {})});
}
function table(s,values,x,y,width,height,widths,font=24){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,values,columnWidths:widths});
  t.borders.assign({fill:'#CCD8DD',width:1,style:'solid'});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2?'#F2F6F7':'#FFFFFF');
    cell.text.style={fontFamily:FONT,fontSize:font,bold:r===0,color:r===0?'#FFFFFF':C.ink};
  }
  return t;
}
function chart(s,type,cats,series,x,y,w,h,{legend=true,labels=false,max=100,size=19}={}){
  const ch=s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories:cats,
    series:series.map(a=>({name:a[0],values:a[1],fill:a[2],line:{fill:a[2],width:3,style:'solid'},marker:{symbol:'circle',size:6,fill:a[2]}})),
    hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:size,fill:C.ink}},
    barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
    xAxis:{textStyle:{fontSize:size,fill:C.ink},majorGridlines:null},
    yAxis:{min:0,max,majorUnit:20,textStyle:{fontSize:17,fill:C.gray},numberFormatCode:'0',majorGridlines:{fill:'#DEE5E8',width:1,style:'solid'}},
    dataLabels:{showValue:labels,position:'outEnd',numberFormatCode:'0.0',textStyle:{fontSize:21,fill:C.ink}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
  applyPresentationChartFont(ch,{fontFamily:FONT,availableFonts:[FONT]});return ch;
}
function legend(s,y=171){
  const items=[['1194M学習',C.blue,55,220],['多様形状学習',C.green,290,245],['Canny＋輪郭別RANSAC',C.orange,550,320],['Zhang 2019型（再現実装）',C.purple,875,370]];
  for(const [t,c,x,w]of items)text(s,t,x,y,w,28,18,c,true);
}

// 数値は現行の固定比較から読み取った成功率。百分率のまま軸へ渡す。
const names=['1194M学習','多様形状学習','Canny＋輪郭別RANSAC','Zhang 2019型（再現実装）'];
const colors=[C.blue,C.green,C.orange,C.purple];
const target=[
 [[100,100,95.54,.89,0],[99.11,98.81,72.92,0,0],[81.25,58.93,9.82,0,0],[79.46,63.39,.89,0,0]],
 [[100,100,100,100,99.70],[99.11,97.62,94.64,96.13,93.45],[81.25,82.14,83.04,76.79,68.75],[79.46,78.57,79.46,80.36,82.14]],
 [[100,100,99.70,98.81,69.64],[99.11,97.92,89.58,66.67,20.54],[81.25,75,52.68,33.93,16.96],[79.46,75,46.43,14.29,9.82]]];
const diverse=[
 [[79.22,79.22,37.66,0,0],[98.05,98.05,68.40,0,0],[76.62,75.97,1.30,0,0],[88.31,88.31,0,0,0]],
 [[79.22,77.27,77.27,80.95,82.68],[98.05,98.05,97.62,96.32,96.75],[76.62,79.22,74.03,74.68,74.03],[88.31,88.31,89.61,86.36,86.36]],
 [[79.22,75.11,70.35,57.14,39.18],[98.05,96.10,90.26,75.97,57.36],[76.62,72.73,67.53,53.25,39.61],[88.31,82.47,75.97,61.69,48.70]]];

export async function build(p){
  let s=page(p,'画像を用いたロケット上段の','進捗報告会');
  // 表紙は主題と研究の焦点に絞る。
  text(s,'PAF内周検出のロバスト性評価',50,165,1175,85,48,C.ink,true);
  text(s,'特定機体の詳細CADに頼らない学習に向けて',55,280,1160,52,30,C.navy);
  text(s,'B4　伊藤弘道',55,483,800,48,30);
  text(s,'2026年10月　進捗報告',55,540,800,40,23,C.gray);
  s.speakerNotes.text='研究題目は第1・3回カメラ班ミーティングを基に、現在の対象固有CADを使わない学習という目的を副題に示す。発表日は未指定のため月のみ。現在の評価入力はCG画像の輝度情報であり、色の効果は検証していない。';

  s=page(p,'発表内容','目次');
  const outline=[['1','研究背景と目的','デブリ除去とPAF内周検出'],['2','これまでの取り組み','CG評価環境と検出手法の比較'],['3','その後の進捗','多様な合成形状による学習と評価'],['4','まとめと今後の課題','有効な条件と残る限界']];
  outline.forEach((a,i)=>{text(s,a[0],70,185+i*100,70,56,38,C.navy,true);text(s,a[1],160,184+i*100,1000,45,30,C.ink,true);text(s,a[2],160,230+i*100,1000,36,23,C.gray);});

  s=page(p,'デブリ除去には安全な接近が必要','研究背景',
  '第1回カメラ班ミーティング p4を現在の研究目的に合わせて構成。背景の根拠：JAXA「スペースデブリの除去」https://www.kenkai.jaxa.jp/crd2/about/ 、JAXA CRD2プロジェクト https://www.ard.jaxa.jp/crd2/project/index.html （2026-10-25参照）。画像は第1回p4の埋め込み画像（©JAXA）を再使用。対象は通信や姿勢制御の協力を得られない非協力物体として扱う。');
  await img(s,TMP+'/第1回　カメラ班ミーティング/image2.jpg',58,188,500,355,'ロケット上段のPAFを示すJAXA画像');
  text(s,'ロケット上段のPAF　©JAXA',65,548,490,30,17,C.gray);
  text(s,'宇宙デブリの衝突は、\nさらに多くの破片を生む',600,188,600,88,29,C.ink,true);
  text(s,'大型デブリを能動的に除去する\n技術の研究が進んでいる',600,300,600,88,28);
  text(s,'相手の協力が得られないため、\nカメラ画像から特徴部位を捉える',600,423,600,90,28);
  bottom(s,'安全な接近・捕獲のために、対象の「どこに近づくか」を認識する');

  s=page(p,'着目する部位：PAFの内周','研究背景',
  '第1回p4・第3回p4。PAF（Payload Attach Fitting）は衛星とロケットを結合する構造。JAXA CRD2プロジェクトはロケット上段PAFへの最終接近実験を記載：https://www.ard.jaxa.jp/crd2/project/index.html 。本研究は画像内の内周楕円を検出する段階に限定し、相対姿勢推定や把持制御を実装したとはしない。右図は第3回p4に含まれる正解内周付きCG画像。');
  text(s,'PAF：衛星とロケットをつなぐ構造',55,181,1100,46,29,C.navy,true);
  text(s,'衛星分離後もロケット上段に残る\nリング状の部位に着目する',55,267,625,91,28);
  text(s,'円形の内周は、斜めから見ると楕円になる\n画像上の位置・形を求める手掛かりとなる',55,393,630,100,27);
  await img(s,src3+'image6.png',760,220,400,335,'正解内周を重ねた1194MのCG');
  text(s,'水色の線：検出対象の内周',760,554,450,32,20,C.navy);
  bottom(s,'本研究の対象は、PAF内周を画像から楕円として検出すること');

  s=page(p,'照明や遮蔽によって内周が見えにくくなる','研究背景',
  '第1回p4、第3回p4・p7の説明と画像を使用。画像は合成された劣化例であり実宇宙での観測画像ではない。白飛び・黒つぶれ・矩形遮蔽を評価対象にする。レンズフレアや迷光は本比較に含めていない。');
  const imgs=['image2.png','image5.png','image4.png','image3.png'];
  const caps=['通常画像','一部が隠れる','白飛び','黒つぶれ'];
  for(let i=0;i<4;i++){text(s,caps[i],58+i*305,190,280,40,28,C.ink,true);await img(s,src3+imgs[i],58+i*305,250,265,265,caps[i]+'のCG例');}
  text(s,'内周の一部が消えると、外周や部品の輪郭と取り違えやすい',55,548,1170,40,28);
  bottom(s,'残った画像情報から、目的の内周を選び出す必要がある');

  s=page(p,'研究目的','研究目的',
  '現在の研究方針：.codex/RESEARCH_HANDOFF.md。対象固有CADを学習に使わず、多様な合成PAFで学んだ検出器を1194Mに適用する。1194M CADは専用学習の比較基準と評価CGの作成に用いる。実画像・任意機体での汎化は未検証。');
  text(s,'特定機体の詳細CADを学習に使わず、',60,202,1150,65,38,C.navy,true);
  text(s,'画像欠損に強いPAF内周検出を目指す',60,273,1150,65,38,C.navy,true);
  text(s,'確認すること',60,386,1130,45,28,C.ink,true);
  text(s,'学習していない形状に適用しても、内周を検出できるか',80,450,1100,45,28);
  text(s,'遮蔽・白飛び・黒つぶれに対して、どこまで性能を保てるか',80,515,1100,45,28);
  note(s,'まずはCG画像で、学習ベース手法と幾何ベース手法を比較する');

  s=page(p,'内周らしい部分を学習し、楕円を当てはめる','研究概要',
  '採用構成：research/paf_generalization_v2/detector.py。可視境界を推定するCNNと中心線化・重み付きRANSACを組み合わせる。画像例：output/paf_generalization_v2/detector_example/prediction.json、prediction_probability.png、prediction_overlay.png。固定test順の先頭通常画像を用いた例で、成功例を選び直したものではない。CNNと幾何の併用自体は第3回からの構成であり、新規提案とはしない。');
  const pred=JSON.parse(await fs.readFile(ROOT+'/output/paf_generalization_v2/detector_example/prediction.json','utf8'));
  const pipe=[['入力画像',pred.image,'① 画像を入力'],['CNNの境界確率',ROOT+'/output/paf_generalization_v2/detector_example/prediction_probability.png','② 内周らしい部分を抽出'],['検出した楕円',ROOT+'/output/paf_generalization_v2/detector_example/prediction_overlay.png','③ 輪郭に楕円を当てはめる']];
  for(let i=0;i<3;i++){text(s,pipe[i][0],65+i*405,185,360,40,27,C.ink,true);await img(s,pipe[i][1],65+i*405,240,330,260,pipe[i][0]);text(s,pipe[i][2],65+i*405,522,360,45,23,C.navy,true);}
  bottom(s,'CNNが内周を選び、幾何処理が残った輪郭から楕円を求める');

  s=page(p,'CGで撮影条件と画像劣化を揃えて評価','これまでの取り組み　第3回まで',
  '第3回カメラ班ミーティング p3・p4・p7。1194M原画像714枚＝102カメラ条件×7照明条件。カメラ条件単位でtrain497、validation105、test112枚に分割する。劣化後は学習1988、validation735、test1456枚。画像は第3回p3のカメラ・照明条件例。');
  await img(s,src3+'image1.png',55,172,605,390,'1194Mのカメラと照明条件の例');
  text(s,'1194Mの簡易CADから714枚を生成',700,196,520,78,27,C.navy,true);
  text(s,'撮影方向・距離・照明を変える',700,310,520,45,27);
  text(s,'学習用とテスト用で\nカメラ条件を分ける',700,389,520,90,27);
  text(s,'同じ画像に各手法を適用する',700,514,520,45,27);
  bottom(s,'「見え方の違い」と「画像欠損への耐性」を同じ条件で比較');

  s=page(p,'先行研究に基づく幾何手法を追加','これまでの取り組み　第4回まで',
  '第4回カメラ班ミーティング p4・p8。正規出典：Limin Zhang, Wang Pan, Xianghua Ma, Real-Time Docking Ring Detection Based on the Geometrical Shape for an On-Orbit Spacecraft, Sensors 2019, 19(23), 5243. DOI:10.3390/s19235243。図は第4回p8から再使用。本比較器は著者コードの移植ではなく、処理構成を参考にしたPython再現実装。実装固有の弧数・組合せ数制限とPAF内周候補選択がある。');
  text(s,'Zhang 2019型（再現実装）',55,178,1150,45,31,C.navy,true);
  await img(s,src4+'image7.png',55,244,1170,280,'入力、Canny、弧分類、三弧統合、最終選択の処理例');
  text(s,'分断された円弧を組み合わせ、同じ楕円に属する候補を探す',55,546,1170,45,27);
  bottom(s,'Canny＋輪郭別RANSACに加え、比較対象を2方式に拡張');

  s=page(p,'既知機体ではCNNの欠損耐性を確認','これまでの取り組み　第4回まで',
  '第4回カメラ班ミーティング p10・p12の報告値：CNN74.5%、Canny＋輪郭別RANSAC46.8%、Zhang 2019型（再現実装）47.6%、1194Mテスト1456枚。ここは当時の設定による結果であり、後半の現行結果とは教師・学習・推論等の設定が異なる。数値差をそのまま改善量と解釈しない。');
  text(s,'1194Mでの検出成功率（当時の設定）',55,176,1100,42,27);
  chart(s,'bar',['CNN','Canny＋輪郭別RANSAC','Zhang 2019型\n（再現実装）'],[['成功率',[74.5,46.8,47.6],C.blue]],55,237,735,330,{legend:false,labels:true,size:19});
  text(s,'画像劣化への有効性は確認',825,259,380,82,29,C.navy,true);
  text(s,'次の問い',825,383,380,40,26,C.ink,true);
  text(s,'学習していない機体に\n変わっても使えるか',825,440,380,90,28);
  note(s,'第4回の報告値。後半の最新結果とは設定が異なるため、数値を直接比較しない。');
  bottom(s,'欠損耐性に加えて、機体形状への依存性を調べる');

  s=page(p,'多様な合成PAFで学習データを作る','その後の進捗',
  'parametric_paf/random_paf/config.json、parametric_paf/output/paf_random_outer_d_v2/manifest.json。上端外径を基準とし、下端外径・高さ・穴径を独立に変える。環状上板、閉じた底、一定厚の側壁を持つ簡易形状で、リブ・台座は付けない。画像は保存済み合成画像の先頭6例。原画像1024枚のうちtrain497、validation105、test154枚を使用する。1194MのCAD・画像・重みを多様形状CNNの学習・validation・初期化に混ぜない。');
  for(let i=0;i<6;i++)await img(s,ROOT+`/parametric_paf/output/paf_random_outer_d_v2/images/paf_${String(i).padStart(4,'0')}.png`,55+(i%3)*210,184+Math.floor(i/3)*195,190,180,'異なる寸法の合成PAF例 '+i);
  text(s,'高さ・胴体の広がり・穴の大きさを変える',720,199,490,90,29,C.navy,true);
  text(s,'特定の機体を再現しない\n簡単な形状を多数生成',720,337,490,90,28);
  text(s,'1194MのCAD・画像は\nこのモデルの学習に使わない',720,465,490,90,27);
  bottom(s,'共通する「内周の見え方」を学べるかを検証する');

  s=page(p,'形状に加え、画像上の見え方も変える','その後の進捗',
  '研究図：output/paf_core_ablation_v2/figures/factorial_views.png/json。固定train先頭paf_0673。投影は内周の大きさ・位置・扁平率を変え、外観は反射色へノイズ・縞・格子を加える。外観変更は濃淡・平均反射色も含むため模様単独の効果とはしない。多様形状CNNは4ビュー各25%を使用。既知の矩形遮蔽を除いた境界を教師にする。露光劣化後の可視性を直接判定した教師ではない。');
  await img(s,core+'figures/factorial_views.png',50,182,1180,379,'同じ形状の投影と外観を別々に変える4例');
  text(s,'投影：大きさ・位置・つぶれ方　　外観：表面の濃淡や模様',55,566,1170,36,25);
  bottom(s,'見えている内周を学習し、一部が隠れる例も学習に含める');

  s=page(p,'学習元の異なるCNNと幾何手法を比較','現在の評価条件',
  report+'。両CNNは境界教師・推論処理・60epoch・学習原画像497枚を共通化。1194M側は投影拡張あり・外観変更なし、多様形状側は投影・外観拡張あり。したがって全学習条件を同一とはしない。testは1194M112原画像×13条件＝1456、多様形状154×13＝2002。CNNは固定3seed平均、幾何は1回評価。成功は正解と推定の楕円内部領域IoUが0.8以上。全画像を分母に含む。');
  table(s,[['比較する方式','学習データ'],['① 1194M学習CNN','1194Mの簡易CAD画像'],['② 多様形状学習CNN','寸法を変えた合成PAF画像'],['③ 幾何ベース2方式','学習なし']],55,180,1170,254,[590,580],25);
  text(s,'評価対象：1194M と 学習に使わない寸法の合成PAF',55,464,1170,40,27,C.navy,true);
  text(s,'成功：正解と推定の楕円領域の重なり（IoU）が0.8以上',55,518,1170,40,26);
  note(s,'通常＋遮蔽・白飛び・黒つぶれの4強度＝13条件。CNNは3回の学習の平均。');
  bottom(s,'①には1194Mの学習経験があり、②にはない状態で比べる');

  s=page(p,'多様形状学習は両対象で幾何手法を上回る','現在の結果　全13条件',
  report+'。CNNは固定3seed平均、幾何は各画像1回。1194M n=1456、多様形状 n=2002。値は全13条件の検出成功率。①81.87/58.16、②71.34/74.84、Canny49.18/53.00、Zhang46.91/61.24。元データの撮影・材質分布、外観拡張の有無も異なるため、形状多様性だけの因果効果とはしない。');
  text(s,'検出成功率（%）',55,175,600,36,24);
  chart(s,'bar',['1194M','多様形状'],names.map((n,i)=>[n,[[81.87,58.16],[71.34,74.84],[49.18,53],[46.91,61.24]][i],colors[i]]),55,223,1170,350,{labels:true,size:18});
  note(s,'黒背景・単一開口のCG画像。CNNは3回の学習の平均。未検出も失敗に含む。');
  bottom(s,'1194Mを学習に使わなくても転移する一方、専用学習との差は残る');

  s=page(p,'1194Mでは遮蔽と強い黒つぶれが課題','現在の結果　学習していない1194Mへの適用',
  report+'。各通常・劣化条件は112画像、CNN3seed平均。各パネルの強度0は同一の通常画像を再表示。ZhangはZhang 2019型（再現実装）。画像劣化強度は各劣化の設定値であり3種類間で等しい物理的強度を意味しない。');
  legend(s);const ttl=['矩形遮蔽','白飛び','黒つぶれ'];
  for(let i=0;i<3;i++){text(s,ttl[i],65+i*398,223,370,40,27,C.ink,true);chart(s,'line',['0','0.25','0.5','0.75','1.0'],target[i].map((v,j)=>[names[j],v,colors[j]]),45+i*400,268,385,285,{legend:false,size:17});}
  note(s,'縦軸：成功率（%）　横軸：劣化強度（0は通常）　CNNは3回の学習の平均');
  bottom(s,'通常99.1%、半画面遮蔽72.9%。強い黒つぶれでは専用学習との差が拡大');

  s=page(p,'多様形状では寸法が変わっても性能を維持','現在の結果　合成形状の未学習寸法への適用',
  report+'。各条件154画像、CNN3seed平均。同じ合成形状族の未学習寸法であり、任意の未知機体ではない。通常98.05%、半画面遮蔽68.40%、全条件74.84%。遮蔽0.75・1.0では全方式0%。');
  legend(s);for(let i=0;i<3;i++){text(s,['矩形遮蔽','白飛び','黒つぶれ'][i],65+i*398,223,370,40,27,C.ink,true);chart(s,'line',['0','0.25','0.5','0.75','1.0'],diverse[i].map((v,j)=>[names[j],v,colors[j]]),45+i*400,268,385,285,{legend:false,size:17});}
  note(s,'縦軸：成功率（%）　横軸：劣化強度（0は通常）　同じ合成形状族の未学習寸法');
  bottom(s,'多様形状学習の利点は残るが、内周の大部分が隠れる条件には弱い');

  s=page(p,'投影と外観を組み合わせると転移性能が向上','要因の検証',
  'output/paf_core_ablation_v2/figures/factorial_success.json、.codex/RESEARCH_HANDOFF.md「全3seedの要因比較」。多様形状学習から1194Mへの全13条件成功率：元43.20、外観のみ56.94、投影のみ38.19、両方69.30%。3seed平均。4条件は共通の固定遮蔽で、採用構成の開口幅遮蔽再抽選を含まない。主比較と別の対応対照。投影に伴って陰影も変化し、外観は反射色の変化も含む。内部表現の依存先や模様だけの因果効果を証明するものではない。');
  text(s,'1194Mへの検出成功率（%）／多様形状で学習',55,174,1160,42,26);
  chart(s,'bar',['拡張なし','外観のみ','投影のみ','投影＋外観'],[['成功率',[43.20,56.94,38.19,69.30],C.green]],55,234,1170,319,{legend:false,labels:true,size:23});
  note(s,'4条件で教師・検出処理・学習時間・遮蔽条件を固定。3回の学習の平均。');
  bottom(s,'形状を増やすだけでなく、内周の見え方と外観の偏りを減らすことが重要');

  s=page(p,'学習を延ばしても、投影だけでは差が残る','最新の追加確認',
  diagnosis+'。固定seed20260907の診断。多様形状学習から1194Mへの全13条件成功率。120epochを連続学習し、各予算内の元validation loss最小重みを選ぶ。60/120の拡張なし43.20/46.22、投影のみ40.80/43.41、投影＋外観70.95/69.78。全て固定遮蔽で、採用構成の開口幅遮蔽は使わない。過去の3seed平均とは別の比較。120epochの範囲での結果であり、全seed・任意の学習時間に一般化しない。');
  text(s,'1194Mへの検出成功率（%）／多様形状で学習',55,174,1160,42,26);
  chart(s,'bar',['拡張なし','投影のみ','投影＋外観'],[['60 epoch',[43.20,40.80,70.95],'#A9C3D1'],['120 epoch',[46.22,43.41,69.78],C.navy]],55,235,1170,319,{labels:true,size:22});
  note(s,'epochは学習データを一巡する回数。この追加確認は1回の学習で比較。');
  bottom(s,'120 epochまででは、投影のみの不利を学習時間だけで説明できない');

  s=page(p,'残る課題と評価の範囲','考察',
  report+' および .codex/RESEARCH_HANDOFF.md。1194M黒つぶれ強度1.0の成功率は専用69.64、多様20.54。主比較の既存testは研究過程で参照済みで盲検ではない。実画像・任意背景・任意機体・対象なし・複数開口は未検証。強度0.75の遮蔽には全被覆や短弧が混在するので、全てを情報消失として除外しない。');
  text(s,'強い黒つぶれ：1194Mで差が残る',55,182,1150,44,28,C.ink,true);
  chart(s,'bar',['1194M学習','多様形状学習'],[['成功率',[69.64,20.54],C.blue]],55,249,550,286,{legend:false,labels:true,size:23});
  text(s,'黒つぶれ強度1.0の成功率（%）',55,542,560,32,18,C.gray);
  text(s,'検証できた範囲',690,250,510,42,29,C.navy,true);
  text(s,'黒背景・単一開口のCG画像\n1194Mと合成形状族',690,316,510,90,27);
  text(s,'実画像や別の機体への適用は未検証\n未参照の評価データも必要',690,450,510,91,26);
  bottom(s,'「任意の機体で使える」と結論づけるには、追加の検証が必要');

  s=page(p,'まとめと今後の予定','まとめ',
  '現在の方針・根拠：.codex/RESEARCH_HANDOFF.md、output/paf_1194m_no_surface_v1/REPORT.md、output/paf_training_diagnosis_v1/REPORT.md。今後の予定は研究課題として提示し、本資料作成に伴う新しい学習や実験を実施したとはしない。');
  text(s,'ここまでに分かったこと',55,184,1170,45,29,C.navy,true);
  text(s,'多様な合成PAFで学習すると、1194Mを学習に含めずに\n比較した幾何2方式を上回る検出性能を得られた',65,248,1150,93,29);
  text(s,'形状の多様化に加え、投影と外観の変動を含めることが有効',65,367,1150,47,27);
  text(s,'今後の予定',55,453,1170,45,29,C.navy,true);
  text(s,'強い黒つぶれ・短い残存輪郭での失敗を整理する\n未参照の形状・実画像で、適用範囲を確認する',65,515,1150,90,28);

  s=page(p,'主比較の学習・評価条件','補足資料',report);
  table(s,[['項目','1194M学習CNN','多様形状学習CNN'],['学習原画像','497枚','497枚'],['学習時の見え方','投影を変更','投影と外観を変更'],['教師・推論','可視境界＋中心線RANSAC','同左'],['学習','60 epoch、3 seed','同左'],['1194Mテスト','112原画像 × 13条件','同左'],['多様形状テスト','154原画像 × 13条件','同左']],55,178,1170,372,[345,410,415],23);
  note(s,'validationは各735例。各学習元のvalidation lossが最小の重みを採用。');
  bottom(s,'学習条件の全てが同一ではなく、外観変更の有無は異なる');

  s=page(p,'基礎比較：学習元をまたぐと性能が低下','補足資料　穴領域CNNによる比較',
  'output/hole_region_comparison_v1/results/REPORT.md、.codex/PROJECT_CONTEXT.md「確認済みの基礎比較結果」。この基礎比較のCNNは完全な穴領域を学び、その外周へ楕円を当てはめる。本文の可視境界CNNとは教師・推論が異なる。CNN3seed平均、成功IoU≥0.8、同じ13条件、全画像を分母に含む。学習元の形状・撮影・材質の差を含むので形状の単独因果効果とはしない。');
  table(s,[['方式','1194M','多様形状'],['穴領域CNN（1194M学習）','67.26%','0.93%'],['穴領域CNN（多様形状学習）','21.93%','71.83%'],['Canny＋輪郭別RANSAC','49.18%','53.00%'],['Zhang 2019型（再現実装）','46.91%','61.24%']],55,193,1170,310,[650,260,260],24);
  text(s,'自分の学習元では高い性能を示すが、別の学習元への適用は弱い',55,530,1170,43,27);
  bottom(s,'本文の境界CNNとは別仕様。学習元への依存を示す基礎比較として掲載');

  s=page(p,'1194M学習でも投影拡張は転移に寄与','補足資料　固定1seedの追加診断',
  diagnosis+'。現行学習で元材質・教師・遮蔽・最適化を固定し、投影拡張だけを変えた比較。多様形状test全13条件は42.91%から56.74%。差13.83ppは表示値同士の差で、未丸め値の差は13.84pp。本文では差の数値は掲示せず値を示す。元1194M内周長径99.57〜143.57px、多様形状test13.93〜88.74px、追加投影24.20〜179.02px。1seedの確認である。');
  text(s,'多様形状への検出成功率（%）／1194Mで学習',55,176,1170,44,27);
  chart(s,'bar',['投影拡張なし','投影拡張あり'],[['成功率',[42.91,56.74],C.blue]],55,243,685,305,{legend:false,labels:true,size:24});
  text(s,'形状は1194Mのまま',785,265,420,44,28,C.navy,true);
  text(s,'内周の大きさ・位置・\nつぶれ方の経験を増やすと、\n多様形状への適用が改善',785,352,420,145,26);
  note(s,'教師・遮蔽・最適化・元材質は共通。固定1seedで確認。');
  bottom(s,'画像上の投影の偏りを減らすことは、専用学習側にも有効');

  s=page(p,'参考文献・結果の出典','補足資料',
  '構成参考：塚田涼太郎＿進捗報告.pdf、岸知輝_進捗報告.pdf（いずれも2025-10-20）。研究内容には転用しない。背景出典は本スライドと各該当スライドのノートを参照。');
  text(s,'背景・先行研究',55,181,1170,40,27,C.navy,true);
  text(s,'JAXA「スペースデブリの除去」「商業デブリ除去実証（CRD2）」\nhttps://www.kenkai.jaxa.jp/crd2/about/\nhttps://www.ard.jaxa.jp/crd2/project/index.html',55,239,1170,110,23);
  text(s,'Zhang, L., Pan, W., Ma, X.  Sensors 2019, 19(23), 5243\nReal-Time Docking Ring Detection Based on the Geometrical Shape\nfor an On-Orbit Spacecraft　DOI: 10.3390/s19235243',55,373,1170,112,22);
  text(s,'研究資料・結果',55,506,1170,38,26,C.navy,true);
  text(s,'第1・3・4回 カメラ班ミーティング資料\n主比較：paf_1194m_no_surface_v1 ／ 要因比較：paf_core_ablation_v2\n追加確認：paf_training_diagnosis_v1 （各REPORT・集計データ）',55,551,1170,105,21);
  return {slideCount:p.slides.items.length,chartSlides:[10,14,15,16,17,18,19,23],tableSlides:[13,21,22]};
}
