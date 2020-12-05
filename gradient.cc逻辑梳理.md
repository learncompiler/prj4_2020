# gradient.cc逻辑梳理

`gradient.cc`中定义了对于求梯度的c++实现，并将其作为一个pass。



`WithGradientType`的作用是通过输入的type，得出输出的type。

`DeGlobal`的作用是，如果一个expression是一个globalVar，那么将它转变为expression。

`UpdateGrad`的作用是更新梯度，即把grad加到传进来的arg的grad部分（一般是pair中的第二个位置）



`Gradient`函数流程：

1. 将输入的expr转为function的类型，并检查参数是否都是TensorTypeNode
2. 返回的function(ret)的主体在body中定义
3. body干的事情：
   1. 通过ReverseAD对输入的function进行操作，得到返回值rev（包含了求导功能之后的function）
   2. 构造参数normal_args,args（存输入的function中的参数，并拓展出一个和参数同一shape的Expr，初始化为全0，存梯度，两者组合成一个Pair）
   3. 调用Call(rev, args)，结果用c表示
   4. 调用init_grad函数初始化梯度（结果对结果求导为1，故初始化为全1）
   5. 通过`ll->Push(Call(RefRead(bp), {}));`来进行梯度回传的一个过程
   6. 从normal_args中取出计算好的梯度值，存到ret中
   7. 通过get_final_result处理tuple输入的情况，获取最终的前向传播结果
   8. 将前向传播结果和梯度计算结果ret组合成一个Pair，返回



ReverseAD做的事情：

1. 对于FunctionNode，递归调用ReverseAD
2. 对于VarNode，如果值没有存在ad_vars里，就计算它的Expr并存到ad_vars里，否则就直接返回ad_vars里存的对应的Expr
3. 对于GlobalVarNode，我们这里应该不涉及，因为我们在调用gradient函数的时候并未传入mod，而访问GlobalVarNode会对mod是否定义进行检查，若未定义则会报错
4. 遇到IfNode，调用If函数即可
5. 遇到ConstantNode，返回该节点以及一个和它同维度的全0量（使用RefCreate）
6. 遇到CallNode，如果是构造或者调函数，就直接递归调用，而如果是需要进行运算（call->op是OpNode），首先计算出前向结果orig_var，再通过GetRev获得返回值ret，最后再通过nbp_body来更新梯度。





