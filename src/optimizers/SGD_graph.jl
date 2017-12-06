type SgdOptimizerGraph
    graph     :: Graph
    base_lr   :: Any
    iter      :: Int

    function SgdOptimizerGraph(graph::Graph; base_lr=(x->0.001))
         return new(graph, base_lr, 1)
    end
 end

 function optimize(opt::SgdOptimizerGraph, XY)

     loss, pred = forward(opt.graph, XY)
     backward(opt.graph)

     for layer ∈ opt.graph.forward_order
         param = getParam(layer)

         if param == nothing
             continue
         end

         grad = getGradient(layer)
         for j = 1:length(grad)
             param[j] -= opt.base_lr(opt.iter) * grad[j] # does not divide batch
         end
         setParam!(layer, param)
     end

     opt.iter += 1
     return loss, pred
 end
