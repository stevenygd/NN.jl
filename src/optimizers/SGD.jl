type SgdOptimizer
    graph     :: Graph
    base_lr   :: Any
    batch_size :: Int
    iter      :: Int

    function SgdOptimizer(graph::Graph, batch_size::Int=1; base_lr=(x->0.001))
         return new(graph, base_lr, batch_size, 1)
    end
 end

 function optimize(opt::SgdOptimizer, XY)

     loss, pred = forward(opt.graph, XY)
     backward(opt.graph)

     for layer ∈ opt.graph.forward_order
         param = getParam(layer)

         if param == nothing
             continue
         end

         grad = getGradient(layer)
         for j = 1:length(grad)
             param[j] -= opt.base_lr(opt.iter) * grad[j] / opt.batch_size 
         end
         setParam!(layer, param)
     end

     opt.iter += 1
     return loss, pred
 end
