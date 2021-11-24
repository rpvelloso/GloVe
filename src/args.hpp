#include <iostream>
#include <functional>
#include <memory>
#include <vector>

int find_arg(const std::string &str, int argc, char **argv)
{
    int i;
    for (i = 1; i < argc; i++)
    {
        auto arg = std::string(argv[i]);
        if (str == arg)
        {
            if (i == argc - 1)
            {
                std::cerr << "No argument given for " << str << std::endl;
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

template<typename T>
void _setValue(const std::string passed_value, T &var) {
    var = passed_value;    
}

template<>
void _setValue<float>(const std::string passed_value, float &var) {
    var = std::stof(passed_value);
}

template<>
void _setValue<int>(const std::string passed_value, int &var) {
    var = std::stoi(passed_value);
}

struct Parameter {
    Parameter(const std::string &key,
        const std::string &param_type,
        const std::string &default_value,
        const std::string &description
    ) : 
    key(key),
    param_type(param_type),
    default_value(default_value),
    description(description) {};

    virtual void setValue(const std::string &passed_value) = 0;
    virtual ~Parameter() = default;

    std::string key;
    std::string param_type;
    std::string default_value;
    std::string description;
    std::string current_value_str;
};

template<typename T>
struct TemplateParameter : public Parameter
{
    virtual ~TemplateParameter() = default;
    TemplateParameter(
        const std::string &key,
        const std::string &param_type,
        const std::string &default_value,
        const std::string &description,
        T &current_value,
        std::function<void(void)> callback = [](){}
    ) : 
    Parameter(key,param_type,default_value,description),
    current_value(current_value),
    callback(callback) {};

    void setValue(const std::string &passed_value) override {
        current_value_str = passed_value;
        _setValue(passed_value, current_value);
        callback();
    };
    T &current_value;
    std::function<void(void)> callback;
};

void load_parameters(std::vector<std::shared_ptr<Parameter>> &parameters, int argc, char **argv) {
    for (auto p: parameters) {
        int i;
        if ((i = find_arg(p->key, argc, argv)) > 0) {
            p->setValue(argv[i + 1]);
        }
    }
}

void print_parameters(std::vector<std::shared_ptr<Parameter>> &parameters) {
    for (auto p: parameters) {
        std::cout << "\t" << p->key << " <" << p->param_type << ">" << std::endl;
        std::cout << "\t\t" << p->description << ". Default: " << p->default_value << std::endl;
    }
}

void print_loaded_parameters(std::vector<std::shared_ptr<Parameter>> &parameters) {
    for (auto p: parameters) {
        auto val = p->current_value_str.empty()?p->default_value:p->current_value_str;
        std::cout << "\t" << p->key << " = " << val << std::endl;
    }
}
