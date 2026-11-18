import pandas as pd,numpy as np
new=pd.read_csv('frozen_eval/new_readouts_test.csv')
L=new.melt(id_vars=['case','unit','dim','family','noise','truth'],value_vars=['TV_refree','coarea_levelsets','s0_contour'],var_name='method',value_name='estimate')
L['error_percent']=100*(L.estimate/L.truth-1)
old=pd.read_csv('/home/claude/frozen/final_pbgia_study/output/geometry_errors_long.csv'); old=old[old.split=='test']
A=pd.concat([L,old[L.columns]])
A['abs']=A.error_percent.abs()
proto=A.groupby(['dim','noise','method','unit','family'])['abs'].mean().reset_index()
S=proto.groupby(['dim','noise','method'])['abs'].mean().unstack('noise').round(3)
print(S.to_string())
print(proto[proto.noise==0].pivot_table(index=['dim','family'],columns='method',values='abs').round(2)[['TV_refree','coarea_levelsets','TV_train_c','selected','Crofton_train_c','marching_cubes_train_c','marching_squares_train_c']].to_string())
proto.to_csv('frozen_eval/prototype_errors.csv',index=False); S.to_csv('frozen_eval/summary.csv')
