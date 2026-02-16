- problem     def _build_train_step_log_dict(self, loss_metrics: LossMetrics) -> Dict:
        act_in = loss_metrics.act_in
        act_out = loss_metrics.act_out
        feature_acts = loss_metrics.feature_acts
        act_pred = loss_metrics.act_pred
        loss = loss_metrics.mse_loss + loss_metrics.l0_loss # TODO, need to change this

        clt_model = self._get_clt()
        dead_features_per_layer = clt_model.get_dead_features().sum(dim=1)
        dead_features_average_count = dead_features_per_layer.float().mean()
with the feature_sharding 

- still there is this   return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/lustre/home/fdraye/.cache/pypoetry/virtualenvs/clt-GwZYPgeg-py3.11/lib/python3.11/site-packages/torch/autograd/graph.py:824: UserWarning: c10d::allreduce_: an autograd kernel was not registered to the Autograd key(s) but we are trying to backprop through it. This may lead to silently incorrect behavior. This behavior is deprecated and will be removed in a future version of PyTorch. If your operator is differentiable, please ensure you have registered an autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, DispatchKey::CompositeImplicitAutograd). If your operator is not differentiable, or to squash this warning and use the previous behavior, please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd. (Triggered internally at /pytorch/torch/csrc/autograd/autograd_not_implemented_fallback.cpp:62.)

- there is also this: /lustre/home/fdraye/.cache/pypoetry/virtualenvs/clt-GwZYPgeg-py3.11/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py:4631: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
