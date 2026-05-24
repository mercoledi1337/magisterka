class Model {
public:
    virtual std::vector<float> forward(const std::vector<float>& input) = 0;

    virtual void backward(const std::vector<float>& target) = 0;

    virtual void update(float lr) = 0;

    virtual void trainStep(const std::vector<float>& x,
                           const std::vector<float>& y,
                           float lr) = 0;

    virtual ~Model() = default;
};