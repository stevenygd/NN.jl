type SgdOptimizerGraph
    graph     :: Graph
    base_lr :: Any
    iter    :: Int
    batch_size  ::  Int

    function SgdOptimizerGraph(graph::Graph, batch_size;  base_lr=(x->0.01))
         return new(net, base_lr, 1, batch_size)
    end
 end

 function optimize(this::SgdOptimizer, batch_X, batch_Y)

     loss, pred = forward(this.graph, batch_X, batch_Y)
     backward(this.graph, batch_Y)

     for i = 1:length(this.graph.forward_order)
         layer = this.graph.forward_order[i]

         param = getParam(layer)
         if param == nothing
             continue
         end

         gradi = getGradient(layer)
         for j = 1:length(gradi)
             param[j] -= this.base_lr(this.iter) * gradi[j] / this.batch_size
         end
         setParam!(layer, param)
     end

     this.iter += 1;
     return loss, pred
 end
