# TI_QEC
### ti_calc的jax 版本：

用jax对ti_clac函数做改写，文件位于TI_QEC_jax文件夹中。
23.11.10：初始的jax版本是ti_decoder_jax.py中的ti_calc_Parallel_jax函数。其调用的蒙卡行走函数中还保留了for循环语句，仅对单步蒙卡行走做了jit编译。
    
23.11.28：优化的jax版ti_calc函数是phase_flip_ti.py文件中的ti_calc_jax, ti_calc_jax_op函数，两个函数调用的初始化、蒙卡行走以及能量积分函数位于phase_flip_jax.py文件。其中调用的mcmove_jax函数已将for循环修改为lax.scan语句，并用jax.jit修饰；ti_calc_jax主函数部分保留for循环语句，方便输出信息监测计算过程。ti_calc_jax_op函数进一步用lax.scan语句替换了主函数的for循环，并用jit修饰。测试结果显示ti_calc_jax, ti_calc_jax_op在本地cpu版本的jax下运行时，相较于主文件夹TI_QEC/ti_decoder中初始并行版本的ti_calc_Parallel函数，速度提高至4倍。
    
23.11.29：由于近期人大组内gpu服务器繁忙，暂未提交至gpu上运算。

23.12.04：TI函数的jax已成功在GPU上测试,其中ti_calc_jax函数可配合生成自旋翻转动画。

### TI函数的原始版本和其他demo一起保留在main_demo文件夹中

### 多自旋编码的计算文件位于Multi_Spin_Coding文件夹中，其中编码函数位于utils_MSC.py文件里。
