include("FCLayer.jl")
import Calculus: check_gradient

using Base.Test

x = [1.,1.]
w = [1. 1.;
     1. 1.;]
l = FCLayer(2,2)
setParam!(l, w)

# Testing forwarding
@test l.W * x == forward(l,x)
@test l.last_input == x
@test l.last_output == [2.;2.]

#Testing back propagation 
loss = [0.;0.]
@test backward(l,loss) == loss
@test l.last_loss == loss
@test gradient(l) == [0. 0.;0. 0.]

x2 = [2.,3.]
loss2 = [0.;1.]

@test l.W * x2 == forward(l,x2)
@test backward(l,loss2) == [1. ; 1.]
@test l.last_loss == loss2
@test gradient(l) == [0. 0.;2. 3.]


w2 = [2. 3.; 3. 2.]

setParam!(l, w2)
@test l.W == w2
@test backward(l,loss2) == [3. ; 2.]
