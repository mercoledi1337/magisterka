class Model {
public:
    virtual std::vector<float> forward(const std::vector<float>&) = 0;
    virtual void backward(const std::vector<float>&) = 0;
    virtual void update(float lr) = 0;
    virtual ~Model() = default;
};