"""診断と全条件比較を、同じ尺度と明示したseed数で図示する。"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from paf_ring_detection.data import read_json
from research.paf_generalization.diagnose import OUT


def configure():
    """日本語を含む注記を再現可能なフォントで描く。"""
    font=Path('C:/Windows/Fonts/meiryo.ttc')
    fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})


def distributions():
    data=read_json(OUT/'diagnosis.json')['distributions']
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for key,label,color in [('diversified/train','多様形状 train：497形状','#007f86'),
                            ('1194m/test','1194M test：112画像','#d86637')]:
        rows=data[key]['rows']
        axes[0].hist([r['major'] for r in rows],bins=np.linspace(0,160,33),weights=np.ones(len(rows))/len(rows),
                     histtype='step',linewidth=2,label=label,color=color)
        axes[1].hist([r['minor']/r['major'] for r in rows],bins=np.linspace(0,1,26),weights=np.ones(len(rows))/len(rows),
                     histtype='step',linewidth=2,label=label,color=color)
    axes[0].set(xlabel='穴の長径（256入力上のpx）',ylabel='各集合内の割合',title='学習・評価の長径範囲は重ならない')
    axes[1].set(xlabel='短径／長径',ylabel='各集合内の割合',title='極端な斜視の楕円も学習範囲外')
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'distribution_gaps.png',dpi=170)
    fig.savefig(OUT/'distribution_gaps.svg')
    plt.close(fig)


def comparison():
    rows=read_json(OUT/'summary.json')['rows']
    names={'cnn_diversified':'既存CNN：多様形状学習','spatial':'位置・サイズ変換',
           'context':'参照範囲拡大','context_spatial':'参照範囲＋位置・サイズ',
           'structure_context_spatial':'境界の強さ・向き', 'geometry':'楕円制約',
           'spatial_geometry':'楕円制約＋位置・サイズ','annular_context_spatial':'内外周の包含関係',
           'interior_context_spatial':'内部構造の乱数化','surface_context_spatial':'表面模様の乱数化',
           'affine_context':'扁平率・回転を含む変換','boundary_spatial':'境界CNN＋RANSAC',
           'boundary_surface':'表面模様＋境界CNN＋RANSAC',
           'surface_ensemble':'表面模様CNNの確率平均',
           'canny':'Canny＋輪郭別RANSAC','zhang2019':'Zhang 2019型（再現実装）',
           'cnn_1194m':'既存CNN：1194M学習'}
    values={(r['dataset'],r['variant']):r for r in rows if r['degradation']=='all'}
    variants=[v for v in names if all((d,v) in values for d in ('1194m','diversified'))]
    fig,axes=plt.subplots(1,2,figsize=(13,1.2+.45*len(variants)),sharey=True)
    labels=[]
    for variant in variants:
        r=values['1194m',variant]
        suffix='幾何：1回' if variant in ('canny','zhang2019') else f"{r['n_seeds']}seed"
        if variant=='surface_ensemble':
            suffix='3モデル統合'
        labels.append(names[variant]+'  ['+suffix+']')
    for ax,dataset,title in zip(axes,('1194m','diversified'),('1194M：1,456画像','多様形状：2,002画像')):
        for i,v in enumerate(variants):
            row=values[dataset,v]
            color='#85929c' if v in ('canny','zhang2019','cnn_1194m') else '#007f86'
            value=row['success_rate']*100
            ax.barh(i,value,color=color,height=.68)
            label_x=max([value]+[100*x for x in row['per_seed'].values()])+1.5
            ax.text(label_x,i,f'{value:.1f}',va='center',fontsize=9)
            if row['n_seeds']>1:
                ax.scatter([100*x for x in row['per_seed'].values()],[i]*row['n_seeds'],color='#152536',s=15,zorder=3)
        ax.set(xlim=(0,105),xlabel='全条件成功率（%）：楕円IoU ≥ 0.8',title=title)
        ax.set_xticks(range(0,101,20)); ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(range(len(variants)),labels)
    axes[0].invert_yaxis()
    fig.suptitle('対象をまたぐ適用の探索結果：棒は平均、点は各seed',fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT/'comparison.png',dpi=170)
    fig.savefig(OUT/'comparison.svg')
    plt.close(fig)


def interventions():
    data=read_json(OUT/'structure_diagnosis.json')['summary']
    bodies=['original','interior_clutter','surface_clutter']
    models={'baseline':'既存CNN（3seed）','context_spatial':'参照範囲＋位置・サイズ（1seed）',
            'interior_context_spatial':'内部構造の学習（1seed）','surface_context_spatial':'表面模様の学習',
            'boundary_spatial':'境界CNN＋RANSAC（1seed）','boundary_surface':'模様＋境界CNN＋RANSAC'}
    lookup={(r['variant'],r['framing'],r['body']):r for r in data}
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    for ax,framing,title in zip(axes,('native','enlarged'),('元の投影サイズ','約3倍に拡大した投影')):
        for name,label in models.items():
            if all((name,framing,b) in lookup for b in bodies):
                if name in ('surface_context_spatial','boundary_surface'):
                    label+=f"（{lookup[name,framing,'original']['n_predictions']//105}seed）"
                ax.plot(range(len(bodies)),[100*lookup[name,framing,b]['success_rate'] for b in bodies],
                        'o-',label=label,linewidth=1.7)
        ax.set_xticks(range(3),['元画像','内部の棒・小環','表面模様'])
        ax.set(title=title,ylim=(0,105),ylabel='通常validation成功率（%）')
        ax.grid(axis='y',alpha=.2)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,fontsize=8,frameon=False)
    fig.suptitle('同じ105形状の対応付き対照：拡大は正解を用いる感度診断',fontsize=12)
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(OUT/'intervention_sensitivity.png',dpi=170)
    fig.savefig(OUT/'intervention_sensitivity.svg')
    plt.close(fig)


def holdout(repeated=False):
    """同じ投影に対する胴構造の変化と欠損耐性を分けて描く。"""
    path=OUT/'family_holdout_v1/summary.json'
    if not path.exists():
        return
    rows=read_json(path)['rows']
    destination=OUT/'family_holdout_v1'
    if repeated:
        destination=destination/'replications'
        if not (destination/'summary.json').exists():
            return
        rows=[r for r in rows if r['method']!='boundary_surface']
        rows.extend({**r,'method':'boundary_surface'} for r in read_json(destination/'summary.json')['rows'])
    names={'cnn_diversified':'既存CNN：3seed平均','surface_context_spatial':'表面模様CNN：3seed平均',
           'surface_ensemble':'表面模様CNN：3モデル統合','boundary_spatial':'境界CNN＋RANSAC：1seed',
           'boundary_surface':'模様＋境界CNN＋RANSAC：1seed',
           'canny':'Canny＋輪郭別RANSAC','zhang2019':'Zhang 2019型（再現実装）'}
    if repeated:
        names['boundary_surface']='模様＋境界CNN＋RANSAC：3seed平均'
    methods=list(names)
    lookup={(r['method'],r['view'],r['degradation']):100*r['success_rate']
            for r in rows if r['family']=='all'}
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.0),gridspec_kw={'width_ratios':[1,1.15]})
    y=np.arange(len(methods))
    for view,offset,label,color in [('control',-.18,'対応する元構造','#aab5bc'),
                                    ('new_family',.18,'未学習の新構造','#007f86')]:
        values=[lookup[m,view,'all'] for m in methods]
        axes[0].barh(y+offset,values,height=.34,label=label,color=color)
    axes[0].set_yticks(y,[names[m] for m in methods]); axes[0].invert_yaxis()
    axes[0].set(xlim=(0,100),xlabel='成功率（%）：4条件平均',title='周辺構造が変わっても内周は同じ')
    axes[0].legend(frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(0,-.18),ncol=2)
    effects=['clean','black_rectangle','sensor_whiteout','sensor_black_crush']
    matrix=np.array([[lookup[m,'new_family',e] for e in effects] for m in methods])
    axes[1].imshow(matrix,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    axes[1].set_xticks(range(4),['通常','半画面遮蔽','白飛び0.5','黒つぶれ0.5'],rotation=20)
    axes[1].set_yticks(y,[]); axes[1].set_title('新構造：劣化別の成功率（%）')
    for i in range(len(methods)):
        for j in range(4):
            axes[1].text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',
                         color='white' if matrix[i,j]>55 else '#152536')
    title=('固定構成の反復確認：3形状族・96形状対／初回評価による構成選択後の3seed比較' if repeated
           else '固定評価：3形状族・96形状対、各方式768推定／既存13条件平均とは別評価')
    fig.suptitle(title,fontsize=12)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(destination/'comparison.png',dpi=170)
    fig.savefig(destination/'comparison.svg')
    plt.close(fig)


if __name__=='__main__':
    configure()
    distributions()
    comparison()
    interventions()
    holdout()
    holdout(repeated=True)
