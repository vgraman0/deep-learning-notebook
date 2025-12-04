# A Mathematical Framework for Transformer Circuits

Notes on Elhage, et al., "A Mathematical Framework for Transformer Circuits", Transformer Circuits Thread, 2021.

Neel Nanda's *A Walkthrough of A Mathematical Framework for Transformer Circuits* ([link](https://www.youtube.com/watch?v=KV5gbOmHbjU&t=1917s)) was also helpful in putting together these notes.

## Transformer architecture

![alt text](image.png)

Start with a token embedding $x_0 = W_Et$.

Each residual block is processed as follows:
1. Multi-head attention: $x_{i+1} = x_i + \sum_{h \in H_i} h(x_i)$.
2. MLP layer: $x_{i+1} = x_{i+1} + m(x_{i+1})$.
3. Final logits produced by applying the unembedding matrix: $T(t) = W_U x_{-1}$.

## Residual stream

The residual stream is a central object of the architecture. There are a few important properties/implications:

### Communication channel analogy
Layers never process the output of the previous layer. Instead, they read from the stream, process information, and write their result back to the stream.
- As a result, layers can interact out of order. The modern can choose which layers it wants to go through and otherwise go through the residual connection.
- Most of the computation the model does goes through a couple of layers in practice. 
- Some paths matter and most paths don't matter. The goal for interpretability is to focus on paths that matter through the residual stream instead of trying to understand the stream itself (which is extremely complicated due to how much composition takes place).

### Linear structure 

The layers only interact with the residual stream in a linear way (either through addition or a linear map).
- The residual stream doesn't have a "priviledged basis". There are no directions where it is naturally aligned. These normally occur when there are non-linear structures to the network (such as applying ReLU). 
- What does have a priviledged basis? The tokens, the vocabulary, the output logits, the attention patterns, the MLP.
- There are some priviledged directions in the residual stream as a result of floating point arithmetic. Similarly, using the Adam optimizer also induces priviledged directions.

### Virtual weights

At layer $A$ of the network:
- We have a writer to the stream through the output matrix $W_O^A$.
- We have a reader from the stream using the input matrix $W_I^A$.

However, because the stream is linear, Layer 10 is looking at the sum of all previous layers. This means it can look at all the previous layers, and look directly at a circuit layer $B$.

The connection is expressed as $W_I^B \cdot W_O^A$, which is what the paper refers to as a virtual weight.

A useful way to think about it is in terms of bandwidtch and subspaces:
- Layer $A$ writes its information into a specific direction (subspace) of the high-dimensional residual stream.
- Layer $B$ also listens to a specific direction (subspace) of the residual stream.
- Bandwidth:
    - The virtual weight will be large when $B$ is listening to the same subspace as $A$.
    - To the contrary, if $B$ is an orthogonal subspace, the virtual weight will be near zero.

## Attention heads

The paper describes the role of attention heads as the way to move information between tokens in the residual stream. The attention layers are the only bits of the transformer that can move information between positions.

1. Compute the value vector for each token $v_i = W_Vx_i$.
2. Compute a result vector by linearly combining vectors according to the attention pattern $r_i = \sum_j A_{i, j} v_j$.
3. Compute the output vector for each head $h(x)_i = W_O r_i$.

![alt text](image-2.png)

Some ideas:
- It is pretty difficult to interpret the value vector since they are the output of $W_{OV} = W_O W_V$. We could apply several operations to $W_O$, $W_V$ that keep the product the same.
    - $W_{OV}$ is a big matrix that has a low-rank factorization (since $H$ is much smaller than $M$).
