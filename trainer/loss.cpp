#include "loss.h"
#include <cmath>
#include <algorithm>

Node* MSELoss::forward(Node* pred, Node* target) {
    Node* neg_target = new MulNode(target, new LeafNode(Tensor({1})));
    neg_target->parents[1]->value.data[0] = -1.0f;
    Node* diff = new AddNode(pred, neg_target);
    return new MulNode(diff, diff);
}

Node* BCELoss::forward(Node* pred, Node* target) {
    Node* log_p = new LogNode(pred);
    Node* term1 = new MulNode(target, log_p);

    Node* one = new LeafNode(Tensor({1}));
    one->value.data[0] = 1.0f;

    Node* one_minus_p = new AddNode(one,
        new MulNode(pred, new LeafNode(Tensor({1}))));
    one_minus_p->parents[1]->parents[1]->value.data[0] = -1.0f;

    Node* log_one_minus_p = new LogNode(one_minus_p);

    Node* one_minus_y = new AddNode(one,
        new MulNode(target, new LeafNode(Tensor({1}))));
    one_minus_y->parents[1]->parents[1]->value.data[0] = -1.0f;

    Node* term2 = new MulNode(one_minus_y, log_one_minus_p);
    Node* sum = new AddNode(term1, term2);

    Node* neg = new MulNode(sum, new LeafNode(Tensor({1})));
    neg->parents[1]->value.data[0] = -1.0f;

    return neg;
}

SoftmaxCENode::SoftmaxCENode(Node* logits, Node* target)
    : Node(Tensor({1})) {

    parents = {logits, target};

    int n = logits->value.size();
    softmax.resize(n);

    float max_logit = logits->value.data[0];
    for (int i = 1; i < n; ++i)
        max_logit = std::max(max_logit, logits->value.data[i]);

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        softmax[i] = std::exp(logits->value.data[i] - max_logit);
        sum += softmax[i];
    }

    for (int i = 0; i < n; ++i)
        softmax[i] /= sum;

    float loss_val = 0.0f;
    for (int i = 0; i < n; ++i)
        if (target->value.data[i] > 0.0f)
            loss_val -= std::log(softmax[i]);

    value.data[0] = loss_val;
}

void SoftmaxCENode::backward() {
    Node* logits = parents[0];
    Node* target = parents[1];

    for (int i = 0; i < logits->value.size(); ++i)
        logits->grad.data[i] += softmax[i] - target->value.data[i];
}

Node* SoftmaxCELoss::forward(Node* logits, Node* target) {
    return new SoftmaxCENode(logits, target);
}
