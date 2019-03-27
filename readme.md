

## 关于模板库

改变自 ECNU WF队伍 F0RE1GNERS 的模板，队伍 [wiki链接](https://acm.ecnu.edu.cn/wiki/index.php?title=ECNU_Foreigners_(2018))（[备用链接](https://eoj.i64d.com/wiki/index.php?title=ECNU_Foreigners_(2018))）。

在基础上改造成自己用的



## 代码特性

+ 优先保证代码简洁和可读性，其次是常数
+ 数据结构使用指针（除了要卡空间的主席树）
+ 代码尽量用 `namespace` 封装
+ 轻度压行，不使用逗号压行
+ 使用 `template` 来复用代码
+ 代码符合 C++11 标准，且至少需要 C++11
+ 系统相关的部分只考虑 *unix

## pandoc编译指令
pandoc -N -s --toc --pdf-engine=xelatex -V CJKmainfont='黑体' -V mainfont='Times New Roman' -V geometry:margin=1in al
l.md -o output.pdf