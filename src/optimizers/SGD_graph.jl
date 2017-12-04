type SgdOptimizerGraph
    graph     :: Graph
    base_lr :: Any

    function SgdOptimizerGraph(graph::Graph; base_lr=(x->0.01))
         return new(graph, base_lr)
    end
 end

 function optimize(opt::SgdOptimizerGraph, XY::Dict{Layer, Array{Float64}})

     loss, pred = forward(opt.graph, XY)
     backward(opt.graph)

     for layer ∈ opt.graph.forward_order
         param = getParam(layer)

         if param == nothing
             continue
         end

         grad = getGradient(layer)
         for j = 1:length(gradi)
             param[j] -= opt.base_lr(opt.iter) * gradi[j] # does not divide batch
         end
         setParam!(layer, param)
     end

     return loss, pred
 end
