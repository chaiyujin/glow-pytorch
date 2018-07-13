import io
import os
import json
import datetime


class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__setitem__("_fixed", False)
        for arg in args:
            if isinstance(arg, dict):
                self.__append_dict(arg)
        self.__append_dict(kwargs)
        self.make_constant()

    def __append_dict(self, dictionary):
        for k in dictionary:
            if isinstance(dictionary[k], dict):
                self.__append_dict(dictionary[k])
            else:
                self.__setitem__(k, dictionary[k])

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise KeyError("[Config]: `{}` is not defined in config file.".format(item))

    def __setattr__(self, attr, value):
        self.__setitem__(attr, value)

    def __setitem__(self, item, value):
        if item.find("_") == 0 and item != "_fixed":
            raise ValueError("[Config]: item should not start with '_'.")
        if item != "_fixed" and self._fixed:
            raise Exception("[Config]: Do not change constant value.")
        if item != "_fixed" and item in self:
            raise KeyError("[Config]: {} appear more than once.".format(item))
        super().__setitem__(item, value)

    def clear(self):
        names = []
        for k in self:
            names.append(k)
        for k in names:
            del self[k]
        self.__setitem__("_fixed", False)

    def make_constant(self):
        self._fixed = True

    def load_from_json(self, file_path):
        with open(file_path, "r") as load_f:
            load_dict = json.load(load_f)
            self.clear()
            self.__append_dict(load_dict)
        self.make_constant()

    def load_from_dict(self, dictionary):
        self.clear()
        self.__append_dict(dictionary)
        self.make_constant()


class JsonConfig(dict):
    """
    The configures will be loaded and dumped as json file.
    The Structure will be maintained as json.
    [TODO]: Some `asserts` can be make by key `__assert__`
    """
    Indent = 2
    def __init__(self, *argv, **kwargs):
        super().__init__()
        super().__setitem__("__name", "default")
        # check input
        assert len(argv) == 0 or len(kwargs) == 0,\
            ("[JsonConfig]: Cannot initialize with" +
             " position parameters (json file or a dict)"+
             " and named parameters (key and values) at the same time.")
        if len(argv) > 0:
            # init from a json or dict
            assert len(argv) == 1, "[JsonConfig]: Need one positional parameters, found two."
            arg = argv[0]
        else:
            arg = kwargs
        # begin initialization
        if isinstance(arg, str):
            super().__setitem__("__name", os.path.splitext(os.path.basename(arg))[0])
            with open(arg, "r") as load_f:
                arg = json.load(load_f)
        if isinstance(arg, dict):
            # case 1: init from dict
            for key in arg:
                value = arg[key]
                if isinstance(value, dict):
                    value = JsonConfig(value)
                super().__setitem__(key, value)
        else:
            raise TypeError("[JsonConfig]: Do not support given input with type {}".format(type(arg)))

    def __setattr__(self, attr, value):
        raise Exception("[JsonConfig]: Cannot set constant key {}".format(attr))

    def __setitem__(self, item, value):
        raise Exception("[JsonConfig]: Cannot set constant key {}".format(item))
    
    def __getattr__(self, attr):
        return super().__getitem__(attr)

    def __str__(self):
        return self.__to_string("", 0)

    def __to_string(self, name, intent):
        ret = " " * intent + name + " {\n"
        for key in self:
            if key.find("__") == 0:
                continue
            value = self[key]
            line = " " * intent
            if isinstance(value, JsonConfig):
                line += value.__to_string(key, intent + JsonConfig.Indent) + "\n"
            else:
                line += " " * JsonConfig.Indent + key + ": " + str(value) + "\n"
            ret += line
        ret += " " * intent + "}"
        return ret

    def __add__(self, b):
        assert isinstance(b, JsonConfig)
        for k in b:
            v = b[k]
            if k in self:
                if isinstance(v, JsonConfig):
                    super().__setitem__(k, self[k] + v)
                else:
                    if k == "__name":
                        super().__setitem__(k, self[k] + "&" + v)
                    else:
                        assert v == self[k], ("[JsonConfig]: Two config conflicts at" +
                            "`{}`, {} != {}".format(
                            k, self[k], v))
            else:
                # new key, directly add
                super().__setitem__(k, v)
        return self

    def date_name(self):
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "").replace(":", "").replace(" ", "_")
        return date + "_" + super().__getitem__("__name") + ".json"

    def dump(self, dir_path, json_name=None):
        if json_name is None:
            json_name = self.date_name()
        json_path = os.path.join(dir_path, json_name)
        with open(json_path, "w") as fout:
            print(str(self))
            json.dump(self.to_dict(), fout, indent=JsonConfig.Indent)

    def to_dict(self):
        ret = {}
        for k in self:
            if k.find("__") == 0:
                continue
            v = self[k]
            if isinstance(v, JsonConfig):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret


if __name__ == "__main__":
    # d = Config()
    # d.load_from_dict({"shit": True})
    # d.load_from_json("hyper/dctts_LJSpeech.json")
    # d.make_constant()
    # print(d)
    # print(d.x)

    config0 = JsonConfig("hparams_json/LJSpeech-1.1.json")
    # config1 = JsonConfig("hyper/dctts_LJSpeech.json")
    config = config0  # + config1
    config.date_name()
    # print(config)
    config.dump(dir_path="hparams_json")
