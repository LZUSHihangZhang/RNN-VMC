// rnn_cpp.cpp — RNN-VMC C++ 加速扩展 (采样 + ln_psi 前向, 无 autograd 需求)
// 构建: torch.utils.cpp_extension.load (见 rnn_cpp_build.py)
// 参数 P 顺序 (torch 张量向量):
//   0 Wu, 1 Wr, 2 Wc1, 3 Wc2, 4 bu, 5 br, 6 bc1, 7 bc2,
//   8 U_re, 9 c_re, 10 U_im, 11 c_im, 12 h0, 13 onehot
#include <torch/extension.h>
#include <vector>

static torch::Tensor gru_step(const torch::Tensor& h, const torch::Tensor& x,
                              const std::vector<torch::Tensor>& P) {
    auto xh = torch::cat({h, x}, 1);
    auto u = torch::sigmoid(torch::addmm(P[4], xh, P[0].t()));     // bu, Wu
    auto r = torch::sigmoid(torch::addmm(P[5], xh, P[1].t()));     // br, Wr
    auto ht = torch::tanh(torch::addmm(P[7], x, P[3].t())
                          + r * torch::addmm(P[6], h, P[2].t()));  // bc2/Wc2, bc1/Wc1
    return (1 - u) * h + u * ht;
}

static void apply_quota(torch::Tensor& p, const torch::Tensor& n_up,
                        const torch::Tensor& n_down, long n_up_t,
                        long n_down_t) {
    p = p.clone();
    p.index_put_({n_down >= (double)n_down_t, 1}, 0.0);
    p.index_put_({n_up >= (double)n_up_t, 0}, 0.0);
    p = p / p.sum(1, true);
}

// --------------------------------------------------------------------- 采样
torch::Tensor sample_cpp(std::vector<torch::Tensor> P, long batch, long L,
                         long n_up_t, long n_down_t, bool sz0) {
    auto opts = P[0].options();
    auto states = torch::zeros({batch, L}, opts);
    auto h = P[12].expand({batch, -1}).clone();            // h0
    auto n_up = torch::zeros({batch}, opts);
    auto n_down = torch::zeros({batch}, opts);
    auto one = torch::ones({batch}, opts);
    auto neg = -one;
    for (long i = 0; i < L; i++) {
        auto p = torch::softmax(2.0 * torch::addmm(P[9], h, P[8].t()), 1);
        if (sz0) apply_quota(p, n_up, n_down, n_up_t, n_down_t);
        auto up = torch::rand({batch}, opts)
                  < p.index({torch::indexing::Slice(), 0});
        states.index_put_({torch::indexing::Slice(), i},
                          torch::where(up, one, neg));
        n_up += up.to(opts.dtype());
        n_down += (~up).to(opts.dtype());
        auto x = P[13].index({(~up).to(torch::kLong)});
        h = gru_step(h, x, P);
    }
    return states;
}

// ------------------------------------------------------------- ln_psi 前向
std::vector<torch::Tensor> lnpsi_forward_cpp(std::vector<torch::Tensor> P,
                                             torch::Tensor states,
                                             bool sz0, long n_up_t,
                                             long n_down_t) {
    auto opts = P[0].options();
    long B = states.size(0), L = states.size(1);
    auto idx = (states < 0).to(torch::kLong);               // 0:+1, 1:-1
    auto onehot = P[13].index({idx});                       // (B,L,2)
    auto h = P[12].expand({B, -1}).clone();
    auto n_up = torch::zeros({B}, opts);
    auto n_down = torch::zeros({B}, opts);
    auto logP = torch::zeros({B}, opts);
    auto phi = torch::zeros({B}, opts);
    auto ar = torch::arange(B, torch::kLong);
    for (long i = 0; i < L; i++) {
        auto x = onehot.index({torch::indexing::Slice(), i,
                               torch::indexing::Slice()});
        auto w_re = torch::addmm(P[9], h, P[8].t());
        auto w_im = torch::addmm(P[11], h, P[10].t());
        auto q = torch::softmax(2.0 * w_re, 1);
        if (sz0) apply_quota(q, n_up, n_down, n_up_t, n_down_t);
        auto col = idx.index({torch::indexing::Slice(), i});
        logP += torch::log(q.index({ar, col}));
        phi += w_im.index({ar, col});
        h = gru_step(h, x, P);
        n_up += (col == 0).to(opts.dtype());
        n_down += (col == 1).to(opts.dtype());
    }
    return {logP, phi};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_cpp", &sample_cpp, "RNN autoregressive sampling (C++)");
    m.def("lnpsi_forward_cpp", &lnpsi_forward_cpp, "RNN ln_psi forward (C++)");
}
