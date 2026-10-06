import os
import yaml
import gc
import torch


def load_config(config_name, config_dir="config"):
    # Nếu config_name là đường dẫn tuyệt đối hoặc đã tồn tại, dùng trực tiếp
    if os.path.isabs(config_name) or os.path.exists(config_name):
        config_path = os.path.abspath(config_name)
    else:
        # Nếu chỉ truyền tên file, tìm trong thư mục config nằm cùng cấp src
        src_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(src_dir)
        config_dir_path = os.path.join(root_dir, "config")
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"
        config_path = os.path.join(config_dir_path, config_name)
        config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if isinstance(config, dict) and "config" in config:
        return config["config"]
    return config


def get_arg_or_config(arg_val, config_val, default_val):
    # Nếu arg_val là một biến kiểu str, kiểm tra nếu là tên biến thì lấy giá trị biến đó
    # Nếu arg_val là None thì lấy config_val, nếu config_val cũng None thì lấy default_val
    if isinstance(arg_val, str):
        # Nếu arg_val là tên biến (ví dụ: "img_size") và biến đó tồn tại trong locals/globals, lấy giá trị biến
        import inspect

        frame = inspect.currentframe().f_back
        if arg_val in frame.f_locals:
            return frame.f_locals[arg_val]
        if arg_val in frame.f_globals:
            return frame.f_globals[arg_val]
        # Nếu không phải tên biến, trả về giá trị string như bình thường
        return arg_val
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default_val


def clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
