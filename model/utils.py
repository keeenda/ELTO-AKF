import numpy as np

def parameter_transform(transform):
    """对传入的关键字参数进行变换。"""
    def _parameter_transform(func):
        def _transformed_parameters_func(*args, **kwargs):
            if isinstance(transform, dict):
                transformed_kwargs = {k: transform[k](kwargs[k]) for k in kwargs
                                      if k in transform and transform[k] is not None}
                kwargs.update(transformed_kwargs)
            elif transform is not None:
                kwargs = {k: transform(kwargs[k]) for k in kwargs}
            return func(*args, **kwargs)
        return _transformed_parameters_func
    return _parameter_transform


def parameter_naming(names):
    """将作为列表传入的参数，根据给定的名字列表，转换为关键字参数。"""
    def _parameter_naming(func):
        def _named_parameters_func(parameters=None, **kwargs):
            if parameters is not None:
                if len(parameters) != len(names):
                    raise Warning("参数数量与参数名字数量不相等。")
                named_parameters = dict(zip(names, parameters))
                kwargs.update(named_parameters)
            return func(**kwargs)
        return _named_parameters_func
    return _parameter_naming


def parameter_arrays(**prefixes):
    """根据前缀将参数分组，形成几个数组/列表。"""
    def _parameter_arrays(func):
        def _parameter_arrays_func(*args, **kwargs):
            # 创建新的参数列表
            param_groups = {group_name: np.array([value for name, value in kwargs.items() if name.startswith(prefix)])
                            for group_name, prefix in prefixes.items()}
            # 删除旧的、未分组的参数
            old_names = [name for name in kwargs for prefix in prefixes.values() if name.startswith(prefix)]
            for name in old_names:
                del kwargs[name]
            # 将分组后的新参数和其余参数一起传入
            return func(*args, **param_groups, **kwargs)
        return _parameter_arrays_func
    return _parameter_arrays


def exception_catcher(exception_type, exception_score):
    """捕获指定类型的异常，并返回一个预设的分数，防止优化过程崩溃。"""
    def _exception_catcher(func):
        def _catched_exception_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type:
                return exception_score
        return _catched_exception_func
    return _exception_catcher