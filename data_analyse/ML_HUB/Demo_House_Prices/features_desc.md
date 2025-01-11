由kimi翻译：

MSSubClass：标识销售中涉及的住宅类型。

        20	1层住宅，1946年及以后建造，各种风格
        30	1层住宅，1945年及以前建造
        40	1层住宅，带完工阁楼，各种年代
        45	1.5层住宅 - 未完工，各种年代
        50	1.5层住宅，完工，各种年代
        60	2层住宅，1946年及以后建造
        70	2层住宅，1945年及以前建造
        75	2.5层住宅，各种年代
        80	分层式或多层住宅
        85	分户式门厅
        90	双拼住宅 - 各种风格和年代
       120	1层规划单元开发（PUD）住宅 - 1946年及以后建造
       150	1.5层PUD住宅 - 各种年代
       160	2层PUD住宅 - 1946年及以后建造
       180	PUD - 多层 - 包括分层式/门厅
       190	2户改造住宅 - 各种风格和年代

MSZoning：标识销售的一般分区分类。
		
       A	农业用地
       C	商业用地
       FV	浮动村住宅
       I	工业用地
       RH	高密度住宅
       RL	低密度住宅
       RP	低密度住宅公园
       RM	中密度住宅
	
LotFrontage：与房产相连的街道线性英尺。

LotArea：地块大小（平方英尺）。

Street：通往房产的道路类型。

       Grvl	碎石路
       Pave	铺砌路
       	
Alley：通往房产的巷道类型。

       Grvl	碎石路
       Pave	铺砌路
       NA	无巷道入口
		
LotShape：房产的大致形状。

       Reg	规则
       IR1	略有不规则
       IR2	中等不规则
       IR3	不规则
       
LandContour：房产的平坦程度。

       Lvl	近乎平坦/水平
       Bnk	坡地 - 从街道标高到建筑的快速且显著上升
       HLS	山坡 - 从一边到另一边的显著坡度
       Low	低洼地
		
Utilities：可用的公用设施类型。
		
       AllPub	所有公共设施（电、气、水、下水道）
       NoSewr	电力、燃气和水（化粪池）
       NoSeWa	仅电力和燃气
       ELO	仅电力
	
LotConfig：地块配置。

       Inside	内部地块
       Corner	角地
       CulDSac	死胡同
       FR2	房产两侧临街
       FR3	房产三侧临街
	
LandSlope：房产的坡度。
		
       Gtl	缓坡
       Mod	中等坡度
       Sev	严重坡度
	
Neighborhood：阿姆斯市范围内的物理位置。

       Blmngtn	布鲁明顿高地
       Blueste	蓝茎
       BrDale	布里亚戴尔
       BrkSide	布鲁克赛德
       ClearCr	清水溪
       CollgCr	学院溪
       Crawfor	克劳福德
       Edwards	爱德华兹
       Gilbert	吉尔伯特
       IDOTRR	爱荷华州交通部和铁路
       MeadowV	草地村
       Mitchel	米切尔
       Names	北阿姆斯
       NoRidge	北岭
       NPkVill	北公园别墅
       NridgHt	北岭高地
       NWAmes	西北阿姆斯
       OldTown	老城区
       SWISU	爱荷华州立大学南侧和西侧
       Sawyer	索耶
       SawyerW	索耶西
       Somerst	索默塞特
       StoneBr	石溪
       Timber	林地
       Veenker	文克尔
			
Condition1：与各种条件的接近程度。
	
       Artery	邻近动脉街道
       Feedr	邻近支线街道
       Norm	正常
       RRNn	距南北向铁路200英尺内
       RRAn	邻近南北向铁路
       PosN	靠近积极的场外特征 - 公园、绿带等
       PosA	邻近积极的场外特征
       RRNe	距东西向铁路200英尺内
       RRAe	邻近东西向铁路
	
Condition2：与各种条件的接近程度（如果存在多个）。
		
       Artery	邻近动脉街道
       Feedr	邻近支线街道
       Norm	正常
       RRNn	距南北向铁路200英尺内
       RRAn	邻近南北向铁路
       PosN	靠近积极的场外特征 - 公园、绿带等
       PosA	邻近积极的场外特征
       RRNe	距东西向铁路200英尺内
       RRAe	邻近东西向铁路
	
BldgType：住宅类型。
		
       1Fam	独栋单户住宅
       2FmCon	双户改造住宅；最初建造时为单户住宅
       Duplx	双拼住宅
       TwnhsE	联排别墅端单元
       TwnhsI	联排别墅内单元
	
HouseStyle：住宅风格。
	
       1Story	一层住宅
       1.5Fin	一又二分之一层住宅：第二层完工
       1.5Unf	一又二分之一层住宅：第二层未完工
       2Story	两层住宅
       2.5Fin	两又二分之一层住宅：第二层完工
       2.5Unf	两又二分之一层住宅：第二层未完工
       SFoyer	分户式门厅
       SLvl	分层式住宅
	
OverallQual：评估房屋的整体材料和完成度。

       10	非常好
       9	优秀
       8	很好
       7	好
       6	高于平均水平
       5	平均水平
       4	低于平均水平
       3	一般
       2	差
       1	非常差
	
OverallCond：评估房屋的整体状况。

       10	非常好
       9	优秀
       8	很好
       7	好
       6	高于平均水平
       5	平均水平
       4	低于平均水平
       3	一般
       2	差
       1	非常差
		
YearBuilt：原始建造日期。

YearRemodAdd：翻新日期（如果未进行翻新或扩建，则与建造日期相同）。

RoofStyle：屋顶类型。

       Flat	平顶
       Gable	人字形屋顶
       Gambrel	盖博式屋顶（谷仓式）
       Hip	坡屋顶
       Mansard	曼萨德屋顶
       Shed	单坡屋顶
		
RoofMatl：屋顶材料。

       ClyTile	粘土瓦或瓷砖
       CompShg	标准（复合）瓦片
       Membran	膜材料
       Metal	金属
       Roll	卷材
       Tar&Grv	砾石与焦油
       WdShake	木瓦
       WdShngl	木瓦片
		
Exterior1st：房屋的外立面覆盖材料。

       AsbShng	石棉瓦
       AsphShn	沥青瓦
       BrkComm	普通砖
       BrkFace	砖面
       CBlock	混凝土砌块
       CemntBd	水泥板
       HdBoard	硬板
       ImStucc	仿灰泥
       MetalSd	金属壁板
       Other	其他
       Plywood	胶合板
       PreCast	预制
       Stone	石材
       Stucco	灰泥
       VinylSd	乙烯基壁板
       Wd Sdng	木壁板
       WdShing	木瓦片
	
Exterior2nd：房屋的外立面覆盖材料（如果有多种材料）。

       AsbShng	石棉瓦
       AsphShn	沥青瓦
       BrkComm	普通砖
       BrkFace	砖面
       CBlock	混凝土砌块
       CemntBd	水泥板
       HdBoard	硬板
       ImStucc	仿灰泥
       MetalSd	金属壁板
       Other	其他
       Plywood	胶合板
       PreCast	预制
       Stone	石材
       Stucco	灰泥
       VinylSd	乙烯基壁板
       Wd Sdng	木壁板
       WdShing	木瓦片
	
MasVnrType：石砌贴面类型。

       BrkCmn	
