# PubTabNet 文档隔离与局部编辑监督准备

日期：2026-10-07，子集选择与模型调用前固定。补H1的表格结构/编辑能力和后续H3/H5的独立校准/评测来源，不是已获得表格模型效果。

## 发布来源及实际清点

IBM原仓库实际链接 `ajimeno/PubTabNet`。固定镜像revision `06963b1af16203f4633c718e2c50109eb57ed658`；11,244,059,914字节tar.gz完整校验SHA256 `90c55e733c85c98edf6d350b77f1e4c23767577555fc932e44ad723674de8d3e`。包内500,777 train、9,115 val、9,138 test图片；标注只含train/val共509,892行。不能用主页的名义568k+替代实际可用分母。

文件名PMC文章ID为可获得的原始document group。实测train/val共有6,105文章，train/test共有6,124，val/test共有216。官方图片split不等于文章隔离，原发布split和本项目role都必须记录。

原始代码库许可说明：IBM标注CDLA-Permissive-1.0，图片属于PMC开放文章及其原许可；镜像卡的CDLA-Sharing-1.0标签与原始说明冲突，保留冲突记录，不推断每幅图的单独版权授权。仅本地研究使用与记录；不把原图/HTML当作许可统一的数据发布。

读取于2026-10-06/07的一手来源：

- <https://github.com/ibm-aur-nlp/PubTabNet>
- <https://github.com/ibm-aur-nlp/PubTabNet/blob/master/LICENSE.md>
- <https://huggingface.co/datasets/ajimeno/PubTabNet/tree/06963b1af16203f4633c718e2c50109eb57ed658>

## 四角色分区

全部出现在published val/test的文章ID从train/model-dev排除。全部published test文章也不能进入calibration/locked-evaluation。剩余published train按`SHA256(20261007-pubtabnet-role:document)`整数mod5=0分model-dev，其余train；剩余val按同hash mod3=0分gate-calibration，其余locked-evaluation。一篇文章只选一张表，所有变体沿用文章角色。

固定配额为128 train、32 model-dev、32 gate-calibration、64 locked-evaluation。按`SHA256(20261007-pubtabnet-order:filename)`顺序取首个满足条件且文章未选过的表：2–12行、4–32 cell、完整规范HTML不超过2,000字符，存在无内嵌格式的数字cell，可以通过现有严格adapter恢复。表格特征/长度过滤先于model，不根据错误或模型质量挑选。将全部published val/test图片字节hash及解码pixel hash作为train/model-dev排除表；四角色间也拒绝精确图/内容重复。

本项目另外已检查的OmniDocBench页面/源图不进入此训练集。PMC文章组只证明本项目角色分离，不证明benchmark预训练无重叠；长度/cell和可执行动作过滤存在明显选择偏差。locked角色的图像与target文件冻结但不做人工查看或模型效果选择，直到模型/阈值独立确定。

## 从标注生成可追溯动作

`html.structure.tokens`与`html.cells`按闭合cell顺序一一对应；文本字符转义，受支持的b/i/sub/sup/br格式保留；覆盖漂移或unsupported inline标签直接记录排除，不静默改词。保留released reference canonical HTML及原始标注hash。

每源至少保持`stop`和一处plain-cell首个数字扰动，target为正确cell文本的`replace_cell`。若存在span>1则构造一个span减1输入，target为恢复两个span属性的`set_span`；若末行不延伸rowspan，构造末行重复输入，target为`delete_row`。所有target均离线来自标注，并必须通过原adapter精确还原reference。它们是受控监督，不是原生OCR或teacher trajectory，不能用来宣称parser迁移。

模型输入仅有源图hash/通用ID/当前HTML及从当前HTML计算的地址，不能包含原始bbox、参考cell答案、正确地址、variant/role真值或official metric。训练target/action另存在监督文件；校准和锁定target不进入optimizer或提示。实际同一源的变体数会因span/row条件变化，报告count和作用域，不把变体数当独立文档数。

## 验证与进入下一步的条件

先执行模块失败重现/通过测试，再冻结并核验archive、annotation、image和prepared manifest hashes；四角色PMC ID交集必须为0；数据文件覆盖、唯一源、目标动作回放和原图bbox是否在图内单列检查。没有启动paid teacher/judge，没有把此前4个demo页面转为train。

这一阶段完成后才有资格实现table SFT/agent编辑监督；native具名table解析器、独立视觉验收、Jev路由与RL比较仍需各自验证。保持/数字/spans/extra-row的人工错误不能代替全量真实错误评估。
