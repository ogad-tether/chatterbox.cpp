#include "supertonic_internal.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace tts_cpp::supertonic::detail {
namespace {

struct f32_tensor { std::vector<float> data; int64_t ne[4] = {1,1,1,1}; };

f32_tensor read_f32(const supertonic_model & m, const std::string & source_name) {
    ggml_tensor * t = require_source_tensor(m, source_name);
    f32_tensor out;
    for (int i = 0; i < 4; ++i) out.ne[i] = t->ne[i];
    out.data.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data.data(), 0, ggml_nbytes(t));
    return out;
}

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865475f)); }
inline float mish(float x) { return x * std::tanh(std::log1pf(std::exp(x))); }

void dense(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
           int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t) oc * IC + ic] * x[ic];
        y[oc] = sum;
    }
}

void dense_matmul_vec(const std::vector<float> & x, const f32_tensor & w, const f32_tensor & b,
                      int IC, int OC, std::vector<float> & y) {
    y.assign(OC, 0.0f);
    for (int oc = 0; oc < OC; ++oc) {
        float sum = b.data[oc];
        for (int ic = 0; ic < IC; ++ic) sum += x[ic] * w.data[(size_t)ic * OC + oc];
        y[oc] = sum;
    }
}

void dense_matmul_time(const std::vector<float> & x, int L, int IC,
                       const f32_tensor & w, const f32_tensor & b,
                       int OC, std::vector<float> & y) {
    y.assign((size_t) L * OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b.data[oc];
            for (int ic = 0; ic < IC; ++ic) sum += x[(size_t)t*IC + ic] * w.data[(size_t)ic*OC + oc];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

void conv1x1(const std::vector<float> & x, int L, int IC,
             const f32_tensor & w, const f32_tensor * b, int OC,
             std::vector<float> & y) {
    y.assign((size_t)L*OC, 0.0f);
    for (int t = 0; t < L; ++t) {
        for (int oc = 0; oc < OC; ++oc) {
            float sum = b ? b->data[oc] : 0.0f;
            for (int ic = 0; ic < IC; ++ic) sum += w.data[(size_t)oc*IC + ic] * x[(size_t)t*IC + ic];
            y[(size_t)t*OC + oc] = sum;
        }
    }
}

ggml_tensor * repeat_like(ggml_context * ctx, ggml_tensor * v, ggml_tensor * like) {
    if (ggml_n_dims(v) == 1 && ggml_n_dims(like) >= 2) {
        if (like->ne[0] == v->ne[0]) v = ggml_reshape_2d(ctx, v, v->ne[0], 1);
        else if (like->ne[1] == v->ne[0]) v = ggml_reshape_2d(ctx, v, 1, v->ne[0]);
    }
    if (!ggml_can_repeat(v, like)) {
        throw std::runtime_error(
            "cannot repeat tensor [" + std::to_string(v->ne[0]) + "," + std::to_string(v->ne[1]) + "," +
            std::to_string(v->ne[2]) + "," + std::to_string(v->ne[3]) + "] to [" +
            std::to_string(like->ne[0]) + "," + std::to_string(like->ne[1]) + "," +
            std::to_string(like->ne[2]) + "," + std::to_string(like->ne[3]) + "]");
    }
    return ggml_repeat(ctx, v, like);
}

ggml_tensor * conv1d_f32(ggml_context * ctx,
                         ggml_tensor * kernel,
                         ggml_tensor * input,
                         int stride,
                         int padding,
                         int dilation) {
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, input, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(ctx, result, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
}

ggml_tensor * edge_clamp_pad_1d(ggml_context * ctx, ggml_tensor * x, int pad_left, int pad_right) {
    const int64_t L = x->ne[0];
    const int64_t C = x->ne[1];
    ggml_tensor * out = x;
    if (pad_left > 0) {
        ggml_tensor * first = ggml_view_2d(ctx, x, 1, C, x->nb[1], 0);
        ggml_tensor * rep = ggml_repeat_4d(ctx, first, pad_left, C, 1, 1);
        out = ggml_concat(ctx, rep, out, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * last = ggml_view_2d(ctx, x, 1, C, x->nb[1], (size_t)(L - 1) * x->nb[0]);
        ggml_tensor * rep = ggml_repeat_4d(ctx, last, pad_right, C, 1, 1);
        out = ggml_concat(ctx, out, rep, 0);
    }
    return out;
}

ggml_tensor * depthwise_same_ggml(ggml_context * ctx,
                                  ggml_tensor * x,
                                  ggml_tensor * w,
                                  ggml_tensor * b,
                                  int dilation) {
    const int K = (int) w->ne[0];
    const int pad_left = ((K - 1) * dilation) / 2;
    const int pad_right = (K - 1) * dilation - pad_left;
    ggml_tensor * padded = edge_clamp_pad_1d(ctx, x, pad_left, pad_right);
    ggml_tensor * new_b = ggml_reshape_4d(ctx, padded, padded->ne[0], 1, padded->ne[1], padded->ne[2]);
    ggml_tensor * im2col = ggml_im2col(ctx, w, new_b, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * y = ggml_mul_mat(ctx, im2col, w);
    y = ggml_reshape_3d(ctx, y, y->ne[0], y->ne[2], 1);
    return ggml_add(ctx, y, repeat_like(ctx, b, y));
}

ggml_tensor * layer_norm_ggml(ggml_context * ctx,
                              ggml_tensor * x,
                              ggml_tensor * g,
                              ggml_tensor * b) {
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
    xt = ggml_norm(ctx, xt, 1e-6f);
    xt = ggml_mul(ctx, xt, repeat_like(ctx, g, xt));
    xt = ggml_add(ctx, xt, repeat_like(ctx, b, xt));
    return ggml_cont(ctx, ggml_permute(ctx, xt, 1, 0, 2, 3));
}

ggml_tensor * dense_matmul_time_ggml(ggml_context * ctx,
                                     ggml_tensor * x,
                                     ggml_tensor * w,
                                     ggml_tensor * b) {
    // Raw ONNX MatMul weights are [IC, OC] in row-major order, while GGML
    // tensors are loaded as ne=[OC, IC].  Make that transpose contiguous, then
    // view it as a Conv1d kernel [K=1, IC, OC] so it can consume the repo's
    // standard time-major activation layout [T, IC].
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w));
    ggml_tensor * kernel = ggml_reshape_3d(ctx, wt, 1, w->ne[1], w->ne[0]);
    ggml_tensor * y = conv1d_f32(ctx, kernel, x, 1, 0, 1);
    if (b) y = ggml_add(ctx, y, repeat_like(ctx, b, y));
    return y;
}

ggml_tensor * vector_convnext_ggml(ggml_context * ctx,
                                   const supertonic_model & model,
                                   const std::string & p,
                                   ggml_tensor * x,
                                   int dilation) {
    ggml_tensor * residual = x;
    ggml_tensor * y = depthwise_same_ggml(ctx, x,
        require_source_tensor(model, p + ".dwconv.weight"),
        require_source_tensor(model, p + ".dwconv.bias"),
        dilation);
    y = layer_norm_ggml(ctx, y,
        require_source_tensor(model, p + ".norm.norm.weight"),
        require_source_tensor(model, p + ".norm.norm.bias"));
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv1.weight"), y, 1, 0, 1);
    y = ggml_add(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".pwconv1.bias"), y));
    y = ggml_gelu_erf(ctx, y);
    y = conv1d_f32(ctx, require_source_tensor(model, p + ".pwconv2.weight"), y, 1, 0, 1);
    y = ggml_add(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".pwconv2.bias"), y));
    y = ggml_mul(ctx, y, repeat_like(ctx, require_source_tensor(model, p + ".gamma"), y));
    return ggml_add(ctx, residual, y);
}

std::vector<float> tensor_to_time_channel(ggml_tensor * t) {
    const int L = (int) t->ne[0];
    const int C = (int) t->ne[1];
    std::vector<float> raw((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
    std::vector<float> out((size_t) L * C);
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < L; ++i) {
            out[(size_t) i * C + c] = raw[(size_t) c * L + i];
        }
    }
    return out;
}

std::vector<float> tensor_raw_f32(ggml_tensor * t) {
    std::vector<float> out((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
    return out;
}

std::vector<float> pack_time_channel_for_ggml(const std::vector<float> & x, int L, int C) {
    std::vector<float> out((size_t)L * C);
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            out[(size_t)c * L + t] = x[(size_t)t * C + c];
        }
    }
    return out;
}

void push_trace(std::vector<supertonic_trace_tensor> & trace,
                const std::string & name,
                int L,
                int C,
                const std::vector<float> & data) {
    trace.push_back({name, {L, C}, data});
}

void depthwise_same(const std::vector<float> & x, int L, int C, const f32_tensor & w,
                    const f32_tensor & b, int K, int dilation, std::vector<float> & y) {
    y.assign((size_t)L*C, 0.0f);
    int pad_left = ((K - 1) * dilation) / 2;
    for (int t = 0; t < L; ++t) {
        for (int c = 0; c < C; ++c) {
            float sum = b.data[c];
            for (int k = 0; k < K; ++k) {
                int st = t + k*dilation - pad_left;
                st = std::max(0, std::min(L - 1, st));
                sum += w.data[(size_t)c*K + k] * x[(size_t)st*C + c];
            }
            y[(size_t)t*C + c] = sum;
        }
    }
}

void layer_norm(std::vector<float> & x, int L, int C, const f32_tensor & g, const f32_tensor & b) {
    for (int t = 0; t < L; ++t) {
        float mean = 0;
        for (int c = 0; c < C; ++c) mean += x[(size_t)t*C+c];
        mean /= (float)C;
        float var = 0;
        for (int c = 0; c < C; ++c) { float d=x[(size_t)t*C+c]-mean; var += d*d; }
        float inv = 1.0f/std::sqrt(var/(float)C + 1e-6f);
        for (int c = 0; c < C; ++c) x[(size_t)t*C+c] = (x[(size_t)t*C+c]-mean)*inv*g.data[c]+b.data[c];
    }
}

void convnext(const supertonic_model & m, const std::string & p, std::vector<float> & x, int L, int C, int dilation) {
    auto dw_w=read_f32(m,p+".dwconv.weight"), dw_b=read_f32(m,p+".dwconv.bias");
    auto ln_g=read_f32(m,p+".norm.norm.weight"), ln_b=read_f32(m,p+".norm.norm.bias");
    auto pw1_w=read_f32(m,p+".pwconv1.weight"), pw1_b=read_f32(m,p+".pwconv1.bias");
    auto pw2_w=read_f32(m,p+".pwconv2.weight"), pw2_b=read_f32(m,p+".pwconv2.bias");
    auto gamma=read_f32(m,p+".gamma");
    std::vector<float> residual=x,y,z;
    depthwise_same(x,L,C,dw_w,dw_b,(int)dw_w.ne[0],dilation,y);
    layer_norm(y,L,C,ln_g,ln_b);
    conv1x1(y,L,C,pw1_w,&pw1_b,(int)pw1_w.ne[2],z);
    for(float &v:z) v=gelu(v);
    conv1x1(z,L,(int)pw1_w.ne[2],pw2_w,&pw2_b,C,y);
    for(size_t i=0;i<x.size();++i){ int c=(int)(i%C); x[i]=residual[i]+gamma.data[c]*y[i]; }
}

std::vector<float> time_embedding(const supertonic_model & m, int current, int total) {
    const int D=64, half=32;
    float t = (float)current / (float)std::max(1,total);
    std::vector<float> emb(D);
    float denom = std::log(10000.0f)/(float)(half-1);
    for(int i=0;i<half;++i){ float f=std::exp((float)i * -denom); float a=t*1000.0f*f; emb[i]=std::sin(a); emb[half+i]=std::cos(a); }
    std::vector<float> h,o;
    dense(emb, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.0.linear.bias"),64,256,h);
    for(float &v:h) v=mish(v);
    dense(h, read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.weight"),
          read_f32(m,"vector_estimator:tts.ttl.vector_field.time_encoder.mlp.2.linear.bias"),256,64,o);
    return o;
}

void apply_rope(const float * theta, std::vector<float> & x, int L, int H, int D) {
    int half = D/2;
    for(int h=0;h<H;++h) for(int t=0;t<L;++t) for(int d=0;d<half;++d) {
        float angle = ((float)t/(float)L)*theta[d];
        float cs=std::cos(angle), sn=std::sin(angle);
        size_t i1=((size_t)t*H+h)*D+d, i2=((size_t)t*H+h)*D+half+d;
        float a=x[i1], b=x[i2];
        x[i1]=a*cs-b*sn; x[i2]=b*cs+a*sn;
    }
}

void rope_attn(const supertonic_model & m, int group, std::vector<float> & x, int L,
               const float * text_emb, int LT, std::vector<float> & out) {
    static const int qids[4]={3101,3146,3191,3236}, kids[4]={3102,3147,3192,3237}, vids[4]={3103,3148,3193,3238}, oids[4]={3110,3155,3200,3245};
    int C=512, A=256, H=4, D=64;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+3)+".attn.";
    std::vector<float> q,k,v;
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    std::vector<float> text_lc((size_t)LT*256);
    for(int t=0;t<LT;++t) for(int c=0;c<256;++c) text_lc[(size_t)t*256+c]=text_emb[(size_t)c*LT+t];
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    dense_matmul_time(text_lc,LT,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    auto theta_t = read_f32(m,"vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
    apply_rope(theta_t.data.data(),q,L,H,D); apply_rope(theta_t.data.data(),k,LT,H,D);
    std::vector<float> attn_out((size_t)L*A,0), scores(LT), probs(LT);
    float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LT;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[((size_t)qi*H+h)*D+d]*k[((size_t)kj*H+h)*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LT;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LT;++kj) sum+=(probs[kj]/den)*v[((size_t)kj*H+h)*D+d]; attn_out[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(attn_out,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

void style_attn(const supertonic_model & m, int group, std::vector<float> & x, int L, const float * style_ttl, std::vector<float> & out) {
    static const int qids[4]={3116,3161,3206,3251}, kids[4]={3117,3162,3207,3252}, vids[4]={3118,3163,3208,3253}, oids[4]={3119,3164,3209,3254};
    int C=512,A=256,H=2,D=128,LC=50;
    std::string base="vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(group*6+5)+".attention.";
    std::vector<float> q,k,v,ctx((size_t)LC*256),kctx((size_t)LC*256);
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) ctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
    auto kconst=read_f32(m,"vector_estimator:/Expand_output_0");
    for(int t=0;t<LC;++t) for(int c=0;c<256;++c) kctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
    dense_matmul_time(x,L,C,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(qids[group])),read_f32(m,base+"W_query.linear.bias"),A,q);
    dense_matmul_time(kctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(kids[group])),read_f32(m,base+"W_key.linear.bias"),A,k);
    for(float &vv:k) vv=std::tanh(vv);
    dense_matmul_time(ctx,LC,256,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(vids[group])),read_f32(m,base+"W_value.linear.bias"),A,v);
    std::vector<float> merged((size_t)L*A,0), scores(LC), probs(LC); float scale=1.0f/16.0f;
    for(int h=0;h<H;++h) for(int qi=0;qi<L;++qi){
        float mx=-INFINITY;
        for(int kj=0;kj<LC;++kj){ float s=0; for(int d=0;d<D;++d) s+=q[(size_t)qi*A+h*D+d]*k[(size_t)kj*A+h*D+d]*scale; scores[kj]=s; mx=std::max(mx,s); }
        float den=0; for(int kj=0;kj<LC;++kj){ probs[kj]=std::exp(scores[kj]-mx); den+=probs[kj]; }
        for(int d=0;d<D;++d){ float sum=0; for(int kj=0;kj<LC;++kj) sum+=(probs[kj]/den)*v[(size_t)kj*A+h*D+d]; merged[(size_t)qi*A+h*D+d]=sum; }
    }
    dense_matmul_time(merged,L,A,read_f32(m,"vector_estimator:onnx::MatMul_"+std::to_string(oids[group])),read_f32(m,base+"out_fc.linear.bias"),C,out);
}

} // namespace

bool supertonic_vector_step_cpu(const supertonic_model & model, const float * noisy_latent,
                                int latent_len, const float * text_emb, int text_len,
                                const float * style_ttl, const float * latent_mask,
                                int current_step, int total_steps,
                                std::vector<float> & next_latent_out, std::string * error) {
    try {
        int L=latent_len,Cin=144,C=512;
        std::vector<float> in((size_t)L*Cin);
        for(int t=0;t<L;++t) for(int c=0;c<Cin;++c) in[(size_t)t*Cin+c]=noisy_latent[(size_t)c*L+t];
        std::vector<float> x;
        conv1x1(in,L,Cin,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_in.net.weight"),nullptr,C,x);
        for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]*=latent_mask[t];
        std::vector<float> te=time_embedding(model,current_step,total_steps);
        static const int time_ids[4]={3095,3140,3185,3230};
        for(int group=0;group<4;++group){
            int ob=group*6;
            int dils[4]={1,2,4,8};
            for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob)+".convnext."+std::to_string(j),x,L,C,dils[j]);
            std::vector<float> tb;
            dense_matmul_vec(te,read_f32(model,"vector_estimator:onnx::MatMul_"+std::to_string(time_ids[group])),
                             read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+1)+".linear.linear.bias"),64,C,tb);
            for(int t=0;t<L;++t) for(int c=0;c<C;++c) x[(size_t)t*C+c]+=tb[c];
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+2)+".convnext.0",x,L,C,1);
            std::vector<float> a; rope_attn(model,group,x,L,text_emb,text_len,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+3)+".norm.norm.bias"));
            convnext(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+4)+".convnext.0",x,L,C,1);
            style_attn(model,group,x,L,style_ttl,a);
            for(size_t i=0;i<x.size();++i) x[i]+=a[i];
            layer_norm(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.weight"),read_f32(model,"vector_estimator:tts.ttl.vector_field.main_blocks."+std::to_string(ob+5)+".norm.norm.bias"));
        }
        for(int j=0;j<4;++j) convnext(model,"vector_estimator:tts.ttl.vector_field.last_convnext.convnext."+std::to_string(j),x,L,C,1);
        std::vector<float> v;
        conv1x1(x,L,C,read_f32(model,"vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),nullptr,Cin,v);
        next_latent_out.assign((size_t)Cin*L,0.0f);
        for(int c=0;c<Cin;++c) for(int t=0;t<L;++t) {
            float vel=v[(size_t)t*Cin+c]*latent_mask[t];
            next_latent_out[(size_t)c*L+t]=noisy_latent[(size_t)c*L+t]+vel/(float)total_steps;
        }
        if(error) error->clear(); return true;
    } catch(const std::exception &e){ if(error)*error=e.what(); return false; }
}

bool supertonic_vector_trace_proj_ggml(const supertonic_model & model,
                                       const float * noisy_latent,
                                       const float * text_emb,
                                       int text_len,
                                       const float * style_ttl,
                                       const float * latent_mask,
                                       int latent_len,
                                       int current_step,
                                       int total_steps,
                                       std::vector<supertonic_trace_tensor> & scalar_trace,
                                       std::vector<supertonic_trace_tensor> & ggml_trace,
                                       std::string * error,
                                       bool include_scalar_trace,
                                       bool include_ggml_trace,
                                       std::vector<float> * next_latent_tc_out) {
    try {
        scalar_trace.clear();
        ggml_trace.clear();
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;
        const int C = 512;
#define PUSH_GGML_TRACE(...) do { if (include_ggml_trace) ggml_trace.push_back(supertonic_trace_tensor __VA_ARGS__); } while (0)
        std::vector<float> in((size_t) L * Cin);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < Cin; ++c) {
                in[(size_t) t * Cin + c] = noisy_latent[(size_t) c * L + t];
            }
        }

        if (include_scalar_trace) {
            push_trace(scalar_trace, "ve_latent_tc", L, Cin, in);

            std::vector<float> proj;
            f32_tensor proj_w = read_f32(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight");
            conv1x1(in, L, Cin, proj_w, nullptr, C, proj);
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) {
                    proj[(size_t) t * C + c] *= latent_mask[t];
                }
            }
            push_trace(scalar_trace, "ve_masked", L, C, proj);

            std::vector<float> block = proj;
            int dils[4] = {1, 2, 4, 8};
            for (int j = 0; j < 4; ++j) {
                convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                         block, L, C, dils[j]);
                push_trace(scalar_trace, "ve_block0_convnext" + std::to_string(j), L, C, block);
            }

            std::vector<float> te = time_embedding(model, current_step, total_steps);
            std::vector<float> tb;
            dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3095"),
                             read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias"),
                             64, C, tb);
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) block[(size_t)t*C+c] += tb[c];
            }
            push_trace(scalar_trace, "ve_time_add0", L, C, block);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0", block, L, C, 1);
            push_trace(scalar_trace, "ve_block2_convnext0", L, C, block);

            const int A = 256;
            std::string base = "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.";
            std::vector<float> q, k, v;
            dense_matmul_time(block, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3101"),
                              read_f32(model, base + "W_query.linear.bias"), A, q);
            std::vector<float> text_lc((size_t) text_len * 256);
            for (int t = 0; t < text_len; ++t) {
                for (int c = 0; c < 256; ++c) {
                    text_lc[(size_t)t * 256 + c] = text_emb[(size_t)c * text_len + t];
                }
            }
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3102"),
                              read_f32(model, base + "W_key.linear.bias"), A, k);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3103"),
                              read_f32(model, base + "W_value.linear.bias"), A, v);
            push_trace(scalar_trace, "ve_attn0_q", L, A, q);
            push_trace(scalar_trace, "ve_attn0_k", text_len, A, k);
            push_trace(scalar_trace, "ve_attn0_v", text_len, A, v);
            auto theta_t = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta_t.data.data(), q, L, 4, 64);
            apply_rope(theta_t.data.data(), k, text_len, 4, 64);
            push_trace(scalar_trace, "ve_attn0_q_rope", L, A, q);
            push_trace(scalar_trace, "ve_attn0_k_rope", text_len, A, k);

            std::vector<float> attn_ctx((size_t)L*A, 0.0f), scores(text_len), probs(text_len);
            const int H = 4, D = 64;
            const float scale = 1.0f / 16.0f;
            for (int h = 0; h < H; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        s += q[((size_t)qi*H+h)*D+d] * k[((size_t)kj*H+h)*D+d] * scale;
                    }
                    scores[kj] = s;
                    mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) {
                    probs[kj] = std::exp(scores[kj] - mx);
                    den += probs[kj];
                }
                for (int d = 0; d < D; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) {
                        sum += (probs[kj] / den) * v[((size_t)kj*H+h)*D+d];
                    }
                    attn_ctx[(size_t)qi*A + h*D + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_attn0_ctx", L, A, attn_ctx);
            std::vector<float> attn_out;
            dense_matmul_time(attn_ctx, L, A, read_f32(model, "vector_estimator:onnx::MatMul_3110"),
                              read_f32(model, base + "out_fc.linear.bias"), C, attn_out);
            push_trace(scalar_trace, "ve_attn0_out", L, C, attn_out);
            std::vector<float> residual = block;
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += attn_out[i];
            push_trace(scalar_trace, "ve_attn0_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.bias"));
            push_trace(scalar_trace, "ve_attn0_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.4.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_block4_convnext0", L, C, residual);

        std::vector<float> style_attn_out;
        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) {
                for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            }
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3116"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3117"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3118"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_style0_q", L, 256, sq);
            push_trace(scalar_trace, "ve_style0_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_style0_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            const int SH=2, SD=128;
            for (int h = 0; h < SH; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < SD; ++d) {
                        score += sq[(size_t)qi*256 + h*SD + d] * sk[(size_t)kj*256 + h*SD + d] * (1.0f/16.0f);
                    }
                    sscores[kj] = score;
                    mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < SD; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*SD + d];
                    smerged[(size_t)qi*256 + h*SD + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_style0_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3119"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias"),
                              C, sout);
            push_trace(scalar_trace, "ve_style0_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_style0_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.bias"));
            push_trace(scalar_trace, "ve_style0_norm", L, C, residual);
        }
        (void) style_attn_out;

        int dils_g1[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.6.convnext." + std::to_string(j),
                     residual, L, C, dils_g1[j]);
            push_trace(scalar_trace, "ve_group1_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3140"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.7.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group1_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.8.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group1_block8_convnext0", L, C, residual);

        {
            const int A1 = 256;
            std::string base1 = "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.";
            std::vector<float> q1, k1, v1;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3146"),
                              read_f32(model, base1 + "W_query.linear.bias"), A1, q1);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3147"),
                              read_f32(model, base1 + "W_key.linear.bias"), A1, k1);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3148"),
                              read_f32(model, base1 + "W_value.linear.bias"), A1, v1);
            push_trace(scalar_trace, "ve_g1_attn_q", L, A1, q1);
            push_trace(scalar_trace, "ve_g1_attn_k", text_len, A1, k1);
            push_trace(scalar_trace, "ve_g1_attn_v", text_len, A1, v1);
            auto theta1 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta1.data.data(), q1, L, 4, 64);
            apply_rope(theta1.data.data(), k1, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g1_attn_q_rope", L, A1, q1);
            push_trace(scalar_trace, "ve_g1_attn_k_rope", text_len, A1, k1);
            std::vector<float> ctx1((size_t)L*A1, 0.0f), scores1(text_len), probs1(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q1[((size_t)qi*4+h)*64+d] * k1[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores1[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs1[kj] = std::exp(scores1[kj]-mx); den += probs1[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs1[kj]/den) * v1[((size_t)kj*4+h)*64+d];
                    ctx1[(size_t)qi*A1 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g1_attn_ctx", L, A1, ctx1);
            std::vector<float> out1;
            dense_matmul_time(ctx1, L, A1, read_f32(model, "vector_estimator:onnx::MatMul_3155"),
                              read_f32(model, base1 + "out_fc.linear.bias"), C, out1);
            push_trace(scalar_trace, "ve_g1_attn_out", L, C, out1);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out1[i];
            push_trace(scalar_trace, "ve_g1_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g1_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.10.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g1_block10_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3161"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3162"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3163"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g1_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g1_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g1_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g1_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3164"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g1_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g1_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g1_style_norm", L, C, residual);
        }

        int dils_g2[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.12.convnext." + std::to_string(j),
                     residual, L, C, dils_g2[j]);
            push_trace(scalar_trace, "ve_group2_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3185"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.13.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group2_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.14.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group2_block14_convnext0", L, C, residual);

        {
            const int A2 = 256;
            std::string base2 = "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.";
            std::vector<float> q2, k2, v2;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3191"),
                              read_f32(model, base2 + "W_query.linear.bias"), A2, q2);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3192"),
                              read_f32(model, base2 + "W_key.linear.bias"), A2, k2);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3193"),
                              read_f32(model, base2 + "W_value.linear.bias"), A2, v2);
            push_trace(scalar_trace, "ve_g2_attn_q", L, A2, q2);
            push_trace(scalar_trace, "ve_g2_attn_k", text_len, A2, k2);
            push_trace(scalar_trace, "ve_g2_attn_v", text_len, A2, v2);
            auto theta2 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta2.data.data(), q2, L, 4, 64);
            apply_rope(theta2.data.data(), k2, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g2_attn_q_rope", L, A2, q2);
            push_trace(scalar_trace, "ve_g2_attn_k_rope", text_len, A2, k2);
            std::vector<float> ctx2((size_t)L*A2, 0.0f), scores2(text_len), probs2(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q2[((size_t)qi*4+h)*64+d] * k2[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores2[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs2[kj] = std::exp(scores2[kj]-mx); den += probs2[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs2[kj]/den) * v2[((size_t)kj*4+h)*64+d];
                    ctx2[(size_t)qi*A2 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g2_attn_ctx", L, A2, ctx2);
            std::vector<float> out2;
            dense_matmul_time(ctx2, L, A2, read_f32(model, "vector_estimator:onnx::MatMul_3200"),
                              read_f32(model, base2 + "out_fc.linear.bias"), C, out2);
            push_trace(scalar_trace, "ve_g2_attn_out", L, C, out2);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out2[i];
            push_trace(scalar_trace, "ve_g2_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g2_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.16.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g2_block16_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3206"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3207"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3208"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g2_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g2_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g2_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g2_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3209"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g2_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g2_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g2_style_norm", L, C, residual);
        }

        int dils_g3[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.18.convnext." + std::to_string(j),
                     residual, L, C, dils_g3[j]);
            push_trace(scalar_trace, "ve_group3_convnext" + std::to_string(j), L, C, residual);
        }
        dense_matmul_vec(te, read_f32(model, "vector_estimator:onnx::MatMul_3230"),
                         read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.19.linear.linear.bias"),
                         64, C, tb);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < C; ++c) residual[(size_t)t*C+c] += tb[c];
        }
        push_trace(scalar_trace, "ve_group3_time_add", L, C, residual);
        convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.20.convnext.0", residual, L, C, 1);
        push_trace(scalar_trace, "ve_group3_block20_convnext0", L, C, residual);

        {
            const int A3 = 256;
            std::string base3 = "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.";
            std::vector<float> q3, k3, v3;
            dense_matmul_time(residual, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3236"),
                              read_f32(model, base3 + "W_query.linear.bias"), A3, q3);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3237"),
                              read_f32(model, base3 + "W_key.linear.bias"), A3, k3);
            dense_matmul_time(text_lc, text_len, 256, read_f32(model, "vector_estimator:onnx::MatMul_3238"),
                              read_f32(model, base3 + "W_value.linear.bias"), A3, v3);
            push_trace(scalar_trace, "ve_g3_attn_q", L, A3, q3);
            push_trace(scalar_trace, "ve_g3_attn_k", text_len, A3, k3);
            push_trace(scalar_trace, "ve_g3_attn_v", text_len, A3, v3);
            auto theta3 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
            apply_rope(theta3.data.data(), q3, L, 4, 64);
            apply_rope(theta3.data.data(), k3, text_len, 4, 64);
            push_trace(scalar_trace, "ve_g3_attn_q_rope", L, A3, q3);
            push_trace(scalar_trace, "ve_g3_attn_k_rope", text_len, A3, k3);
            std::vector<float> ctx3((size_t)L*A3, 0.0f), scores3(text_len), probs3(text_len);
            for (int h = 0; h < 4; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < text_len; ++kj) {
                    float s = 0.0f;
                    for (int d = 0; d < 64; ++d) s += q3[((size_t)qi*4+h)*64+d] * k3[((size_t)kj*4+h)*64+d] * (1.0f/16.0f);
                    scores3[kj] = s; mx = std::max(mx, s);
                }
                float den = 0.0f;
                for (int kj = 0; kj < text_len; ++kj) { probs3[kj] = std::exp(scores3[kj]-mx); den += probs3[kj]; }
                for (int d = 0; d < 64; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < text_len; ++kj) sum += (probs3[kj]/den) * v3[((size_t)kj*4+h)*64+d];
                    ctx3[(size_t)qi*A3 + h*64 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g3_attn_ctx", L, A3, ctx3);
            std::vector<float> out3;
            dense_matmul_time(ctx3, L, A3, read_f32(model, "vector_estimator:onnx::MatMul_3245"),
                              read_f32(model, base3 + "out_fc.linear.bias"), C, out3);
            push_trace(scalar_trace, "ve_g3_attn_out", L, C, out3);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += out3[i];
            push_trace(scalar_trace, "ve_g3_attn_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g3_attn_norm", L, C, residual);
            convnext(model, "vector_estimator:tts.ttl.vector_field.main_blocks.22.convnext.0", residual, L, C, 1);
            push_trace(scalar_trace, "ve_g3_block22_convnext0", L, C, residual);
        }

        {
            std::vector<float> sx = residual;
            for (int t = 0; t < L; ++t) for (int c = 0; c < C; ++c) sx[(size_t)t*C+c] *= latent_mask[t];
            std::vector<float> sq, sk, sv, sctx((size_t)50*256), skctx((size_t)50*256);
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) sctx[(size_t)t*256+c]=style_ttl[(size_t)t*256+c];
            auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
            for (int t = 0; t < 50; ++t) for (int c = 0; c < 256; ++c) skctx[(size_t)t*256+c]=kconst.data[(size_t)t*256+c];
            dense_matmul_time(sx, L, C, read_f32(model, "vector_estimator:onnx::MatMul_3251"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_query.linear.bias"), 256, sq);
            dense_matmul_time(skctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3252"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_key.linear.bias"), 256, sk);
            for (float & vv : sk) vv = std::tanh(vv);
            dense_matmul_time(sctx, 50, 256, read_f32(model, "vector_estimator:onnx::MatMul_3253"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_value.linear.bias"), 256, sv);
            push_trace(scalar_trace, "ve_g3_style_q", L, 256, sq);
            push_trace(scalar_trace, "ve_g3_style_k_tanh", 50, 256, sk);
            push_trace(scalar_trace, "ve_g3_style_v", 50, 256, sv);
            std::vector<float> smerged((size_t)L*256, 0.0f), sscores(50), sprobs(50);
            for (int h = 0; h < 2; ++h) for (int qi = 0; qi < L; ++qi) {
                float mx = -INFINITY;
                for (int kj = 0; kj < 50; ++kj) {
                    float score = 0.0f;
                    for (int d = 0; d < 128; ++d) score += sq[(size_t)qi*256 + h*128 + d] * sk[(size_t)kj*256 + h*128 + d] * (1.0f/16.0f);
                    sscores[kj] = score; mx = std::max(mx, score);
                }
                float den = 0.0f;
                for (int kj = 0; kj < 50; ++kj) { sprobs[kj] = std::exp(sscores[kj]-mx); den += sprobs[kj]; }
                for (int d = 0; d < 128; ++d) {
                    float sum = 0.0f;
                    for (int kj = 0; kj < 50; ++kj) sum += (sprobs[kj]/den) * sv[(size_t)kj*256 + h*128 + d];
                    smerged[(size_t)qi*256 + h*128 + d] = sum;
                }
            }
            push_trace(scalar_trace, "ve_g3_style_ctx", L, 256, smerged);
            std::vector<float> sout;
            dense_matmul_time(smerged, L, 256, read_f32(model, "vector_estimator:onnx::MatMul_3254"),
                              read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias"), C, sout);
            push_trace(scalar_trace, "ve_g3_style_out", L, C, sout);
            for (size_t i = 0; i < residual.size(); ++i) residual[i] += sout[i];
            push_trace(scalar_trace, "ve_g3_style_residual", L, C, residual);
            layer_norm(residual, L, C,
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.weight"),
                       read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.bias"));
            push_trace(scalar_trace, "ve_g3_style_norm", L, C, residual);
        }

        for (int j = 0; j < 4; ++j) {
            convnext(model, "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
                     residual, L, C, 1);
            push_trace(scalar_trace, "ve_last_convnext" + std::to_string(j), L, C, residual);
        }
        std::vector<float> velocity;
        conv1x1(residual, L, C,
                read_f32(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),
                nullptr, Cin, velocity);
        push_trace(scalar_trace, "ve_proj_out", L, Cin, velocity);
        std::vector<float> next_latent((size_t)L * Cin);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < Cin; ++c) {
                float vel = velocity[(size_t)t*Cin+c] * latent_mask[t];
                next_latent[(size_t)t*Cin+c] = noisy_latent[(size_t)c*L+t] + vel / 5.0f;
            }
        }
        push_trace(scalar_trace, "ve_next_latent_tc", L, Cin, next_latent);
        }

        constexpr int MAX_NODES = 2048;
        static size_t buf_size = ggml_tensor_overhead() * MAX_NODES +
                                 ggml_graph_overhead_custom(MAX_NODES, false);
        thread_local std::vector<uint8_t> buf(buf_size);
        ggml_init_params p = { buf_size, buf.data(), true };
        ggml_context * ctx = ggml_init(p);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, Cin);
        ggml_set_name(x, "ve_latent_tc");
        ggml_set_input(x);
        ggml_tensor * mask = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, L);
        ggml_set_name(mask, "ve_latent_mask");
        ggml_set_input(mask);
        ggml_tensor * t_emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_set_name(t_emb, "ve_time_emb");
        ggml_set_input(t_emb);
        ggml_tensor * text_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, text_len, 256);
        ggml_set_name(text_in, "ve_text_lc");
        ggml_set_input(text_in);
        ggml_tensor * q_rope_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, L, 4);
        ggml_set_name(q_rope_in, "ve_q_rope_dlh");
        ggml_set_input(q_rope_in);
        ggml_tensor * k_rope_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(k_rope_in, "ve_k_rope_dlh");
        ggml_set_input(k_rope_in);
        ggml_tensor * v_rope_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(v_rope_in, "ve_v_dlh");
        ggml_set_input(v_rope_in);

        ggml_tensor * y = conv1d_f32(ctx, require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_in.net.weight"), x, 1, 0, 1);
        ggml_tensor * masked = ggml_mul(ctx, y, repeat_like(ctx, mask, y));
        ggml_set_name(masked, "ve_masked");
        ggml_set_output(masked);
        ggml_build_forward_expand(gf, masked);

        ggml_tensor * cur = masked;
        int dils_ggml[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            cur = vector_convnext_ggml(ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.0.convnext." + std::to_string(j),
                cur, dils_ggml[j]);
            const std::string name = "ve_block0_convnext" + std::to_string(j);
            ggml_set_name(cur, name.c_str());
            ggml_set_output(cur);
            ggml_build_forward_expand(gf, cur);
        }

        ggml_tensor * t_proj = ggml_mul_mat(ctx,
            ggml_cont(ctx, ggml_transpose(ctx, require_source_tensor(model, "vector_estimator:onnx::MatMul_3095"))),
            ggml_reshape_2d(ctx, t_emb, 64, 1));
        t_proj = ggml_add(ctx, t_proj,
            ggml_reshape_2d(ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.1.linear.linear.bias"),
                C, 1));
        cur = ggml_add(ctx, cur, repeat_like(ctx, t_proj, cur));
        ggml_set_name(cur, "ve_time_add0");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);

        cur = vector_convnext_ggml(ctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.2.convnext.0",
            cur, 1);
        ggml_set_name(cur, "ve_block2_convnext0");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        ggml_tensor * q_t = dense_matmul_time_ggml(ctx, cur,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3101"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_query.linear.bias"));
        ggml_set_name(q_t, "ve_attn0_q");
        ggml_set_output(q_t);
        ggml_build_forward_expand(gf, q_t);
        ggml_tensor * k_t = dense_matmul_time_ggml(ctx, text_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3102"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_key.linear.bias"));
        ggml_set_name(k_t, "ve_attn0_k");
        ggml_set_output(k_t);
        ggml_build_forward_expand(gf, k_t);
        ggml_tensor * v_t = dense_matmul_time_ggml(ctx, text_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3103"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.W_value.linear.bias"));
        ggml_set_name(v_t, "ve_attn0_v");
        ggml_set_output(v_t);
        ggml_build_forward_expand(gf, v_t);

        ggml_tensor * attn = ggml_flash_attn_ext(ctx, q_rope_in, k_rope_in, v_rope_in,
                                                 nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        attn = ggml_reshape_2d(ctx, attn, 256, L);
        ggml_tensor * attn_tc = ggml_cont(ctx, ggml_transpose(ctx, attn));
        ggml_set_name(attn_tc, "ve_attn0_ctx");
        ggml_set_output(attn_tc);
        ggml_build_forward_expand(gf, attn_tc);
        ggml_tensor * attn_out_t = dense_matmul_time_ggml(ctx, attn_tc,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3110"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.out_fc.linear.bias"));
        ggml_set_name(attn_out_t, "ve_attn0_out");
        ggml_set_output(attn_out_t);
        ggml_build_forward_expand(gf, attn_out_t);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!allocr) throw std::runtime_error("ggml_gallocr_new failed");
        if (!ggml_gallocr_reserve(allocr, gf)) {
            ggml_gallocr_free(allocr);
            throw std::runtime_error("ggml_gallocr_reserve failed");
        }
        ggml_gallocr_alloc_graph(allocr, gf);

        ggml_backend_tensor_set(x, noisy_latent, 0, (size_t) L * Cin * sizeof(float));
        ggml_backend_tensor_set(mask, latent_mask, 0, (size_t) L * sizeof(float));
        std::vector<float> te_host = time_embedding(model, current_step, total_steps);
        ggml_backend_tensor_set(t_emb, te_host.data(), 0, te_host.size() * sizeof(float));
        std::vector<float> text_lc_host((size_t) text_len * 256);
        for (int c = 0; c < 256; ++c) {
            for (int t = 0; t < text_len; ++t) {
                text_lc_host[(size_t)c * text_len + t] = text_emb[(size_t)c * text_len + t];
            }
        }
        ggml_backend_tensor_set(text_in, text_lc_host.data(), 0, text_lc_host.size() * sizeof(float));
        supertonic_graph_compute(model, gf);

        PUSH_GGML_TRACE({"ve_latent_tc", {L, Cin}, in});
        PUSH_GGML_TRACE({"ve_masked", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_masked"))});
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_block0_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_time_add0", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_time_add0"))});
        std::vector<float> block2_ggml = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_block2_convnext0"));
        PUSH_GGML_TRACE({"ve_block2_convnext0", {L, C}, block2_ggml});
        std::vector<float> q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_q"));
        std::vector<float> k_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_k"));
        std::vector<float> v_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_v"));
        PUSH_GGML_TRACE({"ve_attn0_q", {L, 256}, q_out});
        PUSH_GGML_TRACE({"ve_attn0_k", {text_len, 256}, k_out});
        PUSH_GGML_TRACE({"ve_attn0_v", {text_len, 256}, v_out});
        f32_tensor theta = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta.data.data(), q_out, L, 4, 64);
        apply_rope(theta.data.data(), k_out, text_len, 4, 64);
        std::vector<float> q_dlh((size_t)64*L*4), k_dlh((size_t)64*text_len*4), v_dlh((size_t)64*text_len*4);
        for (int h = 0; h < 4; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 64; ++d) {
                q_dlh[(size_t)d + 64*((size_t)t + (size_t)L*h)] = q_out[(size_t)t*256 + h*64 + d];
            }
            for (int t = 0; t < text_len; ++t) for (int d = 0; d < 64; ++d) {
                k_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = k_out[(size_t)t*256 + h*64 + d];
                v_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = v_out[(size_t)t*256 + h*64 + d];
            }
        }
        ggml_backend_tensor_set(q_rope_in, q_dlh.data(), 0, q_dlh.size() * sizeof(float));
        ggml_backend_tensor_set(k_rope_in, k_dlh.data(), 0, k_dlh.size() * sizeof(float));
        ggml_backend_tensor_set(v_rope_in, v_dlh.data(), 0, v_dlh.size() * sizeof(float));
        supertonic_graph_compute(model, gf);
        PUSH_GGML_TRACE({"ve_attn0_q_rope", {L, 256}, q_out});
        PUSH_GGML_TRACE({"ve_attn0_k_rope", {text_len, 256}, k_out});
        PUSH_GGML_TRACE({"ve_attn0_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_ctx"))});
        std::vector<float> attn_out_ggml = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_out"));
        PUSH_GGML_TRACE({"ve_attn0_out", {L, C}, attn_out_ggml});

        constexpr int RES_NODES = 128;
        static size_t res_buf_size = ggml_tensor_overhead() * RES_NODES +
                                     ggml_graph_overhead_custom(RES_NODES, false);
        thread_local std::vector<uint8_t> res_buf(res_buf_size);
        ggml_init_params rp = { res_buf_size, res_buf.data(), true };
        ggml_context * rctx = ggml_init(rp);
        ggml_cgraph * rgf = ggml_new_graph_custom(rctx, RES_NODES, false);
        ggml_tensor * lhs_in = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(lhs_in, "res_lhs"); ggml_set_input(lhs_in);
        ggml_tensor * rhs_in = ggml_new_tensor_2d(rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(rhs_in, "res_rhs"); ggml_set_input(rhs_in);
        ggml_tensor * res = ggml_add(rctx, lhs_in, rhs_in);
        ggml_set_name(res, "ve_attn0_residual"); ggml_set_output(res);
        ggml_build_forward_expand(rgf, res);
        ggml_tensor * norm = layer_norm_ggml(rctx, res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.norm.norm.bias"));
        ggml_set_name(norm, "ve_attn0_norm"); ggml_set_output(norm);
        ggml_build_forward_expand(rgf, norm);
        ggml_tensor * post = vector_convnext_ggml(rctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.4.convnext.0",
            norm, 1);
        ggml_set_name(post, "ve_block4_convnext0"); ggml_set_output(post);
        ggml_build_forward_expand(rgf, post);
        ggml_gallocr_t rallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!rallocr) throw std::runtime_error("ggml_gallocr_new residual failed");
        if (!ggml_gallocr_reserve(rallocr, rgf)) {
            ggml_gallocr_free(rallocr);
            throw std::runtime_error("ggml_gallocr_reserve residual failed");
        }
        ggml_gallocr_alloc_graph(rallocr, rgf);
        std::vector<float> lhs_raw = pack_time_channel_for_ggml(block2_ggml, L, C);
        std::vector<float> rhs_raw = pack_time_channel_for_ggml(attn_out_ggml, L, C);
        ggml_backend_tensor_set(lhs_in, lhs_raw.data(), 0, lhs_raw.size() * sizeof(float));
        ggml_backend_tensor_set(rhs_in, rhs_raw.data(), 0, rhs_raw.size() * sizeof(float));
        supertonic_graph_compute(model, rgf);
        PUSH_GGML_TRACE({"ve_attn0_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(rgf, "ve_attn0_residual"))});
        PUSH_GGML_TRACE({"ve_attn0_norm", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(rgf, "ve_attn0_norm"))});
        std::vector<float> post_ggml = tensor_to_time_channel(ggml_graph_get_tensor(rgf, "ve_block4_convnext0"));
        PUSH_GGML_TRACE({"ve_block4_convnext0", {L, C}, post_ggml});
        ggml_gallocr_free(rallocr);

        constexpr int STYLE_NODES = 256;
        static size_t style_buf_size = ggml_tensor_overhead() * STYLE_NODES +
                                       ggml_graph_overhead_custom(STYLE_NODES, false);
        thread_local std::vector<uint8_t> style_buf(style_buf_size);
        ggml_init_params sp = { style_buf_size, style_buf.data(), true };
        ggml_context * sctx = ggml_init(sp);
        ggml_cgraph * sgf = ggml_new_graph_custom(sctx, STYLE_NODES, false);
        ggml_tensor * style_x = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, L, C);
        ggml_set_name(style_x, "style_x"); ggml_set_input(style_x);
        ggml_tensor * style_v_in = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(style_v_in, "style_ttl_lc"); ggml_set_input(style_v_in);
        ggml_tensor * kctx_in = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(kctx_in, "style_kctx_lc"); ggml_set_input(kctx_in);
        ggml_tensor * style_q_dlh = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 128, L, 2);
        ggml_set_name(style_q_dlh, "style_q_dlh"); ggml_set_input(style_q_dlh);
        ggml_tensor * style_k_dlh = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(style_k_dlh, "style_k_dlh"); ggml_set_input(style_k_dlh);
        ggml_tensor * style_v_dlh = ggml_new_tensor_3d(sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(style_v_dlh, "style_v_dlh"); ggml_set_input(style_v_dlh);

        ggml_tensor * sq = dense_matmul_time_ggml(sctx, style_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3116"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_query.linear.bias"));
        ggml_tensor * sk = dense_matmul_time_ggml(sctx, kctx_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3117"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_key.linear.bias"));
        sk = ggml_tanh(sctx, sk);
        ggml_tensor * sv = dense_matmul_time_ggml(sctx, style_v_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3118"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.W_value.linear.bias"));
        ggml_set_name(sq, "ve_style0_q"); ggml_set_output(sq); ggml_build_forward_expand(sgf, sq);
        ggml_set_name(sk, "ve_style0_k_tanh"); ggml_set_output(sk); ggml_build_forward_expand(sgf, sk);
        ggml_set_name(sv, "ve_style0_v"); ggml_set_output(sv); ggml_build_forward_expand(sgf, sv);
        ggml_tensor * sctx_attn = ggml_flash_attn_ext(sctx, style_q_dlh, style_k_dlh, style_v_dlh, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        sctx_attn = ggml_reshape_2d(sctx, sctx_attn, 256, L);
        ggml_tensor * sctx_tc = ggml_cont(sctx, ggml_transpose(sctx, sctx_attn));
        ggml_set_name(sctx_tc, "ve_style0_ctx"); ggml_set_output(sctx_tc);
        ggml_build_forward_expand(sgf, sctx_tc);
        ggml_tensor * sout = dense_matmul_time_ggml(sctx, sctx_tc,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3119"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias"));
        ggml_set_name(sout, "ve_style0_out"); ggml_set_output(sout);
        ggml_build_forward_expand(sgf, sout);
        ggml_tensor * sres = ggml_add(sctx, style_x, sout);
        ggml_set_name(sres, "ve_style0_residual"); ggml_set_output(sres);
        ggml_build_forward_expand(sgf, sres);
        ggml_tensor * snorm = layer_norm_ggml(sctx, sres,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.bias"));
        ggml_set_name(snorm, "ve_style0_norm"); ggml_set_output(snorm);
        ggml_build_forward_expand(sgf, snorm);

        ggml_gallocr_t sallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!sallocr) throw std::runtime_error("ggml_gallocr_new style failed");
        if (!ggml_gallocr_reserve(sallocr, sgf)) {
            ggml_gallocr_free(sallocr);
            throw std::runtime_error("ggml_gallocr_reserve style failed");
        }
        ggml_gallocr_alloc_graph(sallocr, sgf);
        std::vector<float> style_x_raw = pack_time_channel_for_ggml(post_ggml, L, C);
        std::vector<float> style_v_raw((size_t)50*256), kctx_raw((size_t)50*256);
        auto kconst = read_f32(model, "vector_estimator:/Expand_output_0");
        for (int c = 0; c < 256; ++c) for (int t = 0; t < 50; ++t) {
            style_v_raw[(size_t)c * 50 + t] = style_ttl[(size_t)t * 256 + c];
            kctx_raw[(size_t)c * 50 + t] = kconst.data[(size_t)t * 256 + c];
        }
        ggml_backend_tensor_set(style_x, style_x_raw.data(), 0, style_x_raw.size() * sizeof(float));
        ggml_backend_tensor_set(style_v_in, style_v_raw.data(), 0, style_v_raw.size() * sizeof(float));
        ggml_backend_tensor_set(kctx_in, kctx_raw.data(), 0, kctx_raw.size() * sizeof(float));
        supertonic_graph_compute(model, sgf);
        std::vector<float> sq_out = tensor_to_time_channel(ggml_graph_get_tensor(sgf, "ve_style0_q"));
        std::vector<float> sk_out = tensor_to_time_channel(ggml_graph_get_tensor(sgf, "ve_style0_k_tanh"));
        std::vector<float> sv_out = tensor_to_time_channel(ggml_graph_get_tensor(sgf, "ve_style0_v"));
        PUSH_GGML_TRACE({"ve_style0_q", {L, 256}, sq_out});
        PUSH_GGML_TRACE({"ve_style0_k_tanh", {50, 256}, sk_out});
        PUSH_GGML_TRACE({"ve_style0_v", {50, 256}, sv_out});
        std::vector<float> sq_dlh((size_t)128*L*2), sk_dlh((size_t)128*50*2), sv_dlh((size_t)128*50*2);
        for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 128; ++d) {
                sq_dlh[(size_t)d + 128*((size_t)t + (size_t)L*h)] = sq_out[(size_t)t*256 + h*128 + d];
            }
            for (int t = 0; t < 50; ++t) for (int d = 0; d < 128; ++d) {
                sk_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = sk_out[(size_t)t*256 + h*128 + d];
                sv_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = sv_out[(size_t)t*256 + h*128 + d];
            }
        }
        ggml_backend_tensor_set(style_q_dlh, sq_dlh.data(), 0, sq_dlh.size() * sizeof(float));
        ggml_backend_tensor_set(style_k_dlh, sk_dlh.data(), 0, sk_dlh.size() * sizeof(float));
        ggml_backend_tensor_set(style_v_dlh, sv_dlh.data(), 0, sv_dlh.size() * sizeof(float));
        supertonic_graph_compute(model, sgf);
        PUSH_GGML_TRACE({"ve_style0_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(sgf, "ve_style0_ctx"))});
        std::vector<float> style_ctx_ggml = tensor_to_time_channel(ggml_graph_get_tensor(sgf, "ve_style0_ctx"));
        constexpr int STYLE_RES_NODES = 128;
        static size_t style_res_buf_size = ggml_tensor_overhead() * STYLE_RES_NODES +
                                           ggml_graph_overhead_custom(STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> style_res_buf(style_res_buf_size);
        ggml_init_params srp = { style_res_buf_size, style_res_buf.data(), true };
        ggml_context * srctx = ggml_init(srp);
        ggml_cgraph * srgf = ggml_new_graph_custom(srctx, STYLE_RES_NODES, false);
        ggml_tensor * style_ctx_in = ggml_new_tensor_2d(srctx, GGML_TYPE_F32, L, 256);
        ggml_set_name(style_ctx_in, "style_ctx_in"); ggml_set_input(style_ctx_in);
        ggml_tensor * style_lhs_in = ggml_new_tensor_2d(srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(style_lhs_in, "style_lhs_in"); ggml_set_input(style_lhs_in);
        ggml_tensor * style_out = dense_matmul_time_ggml(srctx, style_ctx_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3119"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.attention.out_fc.linear.bias"));
        ggml_set_name(style_out, "ve_style0_out"); ggml_set_output(style_out);
        ggml_build_forward_expand(srgf, style_out);
        ggml_tensor * style_res = ggml_add(srctx, style_lhs_in, style_out);
        ggml_set_name(style_res, "ve_style0_residual"); ggml_set_output(style_res);
        ggml_build_forward_expand(srgf, style_res);
        ggml_tensor * style_norm = layer_norm_ggml(srctx, style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.5.norm.norm.bias"));
        ggml_set_name(style_norm, "ve_style0_norm"); ggml_set_output(style_norm);
        ggml_build_forward_expand(srgf, style_norm);
        ggml_gallocr_t srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!srallocr) throw std::runtime_error("ggml_gallocr_new style residual failed");
        if (!ggml_gallocr_reserve(srallocr, srgf)) {
            ggml_gallocr_free(srallocr);
            throw std::runtime_error("ggml_gallocr_reserve style residual failed");
        }
        ggml_gallocr_alloc_graph(srallocr, srgf);
        std::vector<float> style_ctx_raw = pack_time_channel_for_ggml(style_ctx_ggml, L, 256);
        std::vector<float> style_lhs_raw = pack_time_channel_for_ggml(post_ggml, L, C);
        ggml_backend_tensor_set(style_ctx_in, style_ctx_raw.data(), 0, style_ctx_raw.size()*sizeof(float));
        ggml_backend_tensor_set(style_lhs_in, style_lhs_raw.data(), 0, style_lhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, srgf);
        PUSH_GGML_TRACE({"ve_style0_out", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(srgf, "ve_style0_out"))});
        PUSH_GGML_TRACE({"ve_style0_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(srgf, "ve_style0_residual"))});
        std::vector<float> style_norm_ggml = tensor_to_time_channel(ggml_graph_get_tensor(srgf, "ve_style0_norm"));
        PUSH_GGML_TRACE({"ve_style0_norm", {L, C}, style_norm_ggml});
        ggml_gallocr_free(srallocr);

        constexpr int G1_NODES = 512;
        static size_t g1_buf_size = ggml_tensor_overhead() * G1_NODES +
                                    ggml_graph_overhead_custom(G1_NODES, false);
        thread_local std::vector<uint8_t> g1_buf(g1_buf_size);
        ggml_init_params g1p = { g1_buf_size, g1_buf.data(), true };
        ggml_context * g1ctx = ggml_init(g1p);
        ggml_cgraph * g1gf = ggml_new_graph_custom(g1ctx, G1_NODES, false);
        ggml_tensor * g1_in = ggml_new_tensor_2d(g1ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_in, "g1_in"); ggml_set_input(g1_in);
        ggml_tensor * g1_temb = ggml_new_tensor_1d(g1ctx, GGML_TYPE_F32, 64);
        ggml_set_name(g1_temb, "g1_temb"); ggml_set_input(g1_temb);
        ggml_tensor * g1cur = g1_in;
        int dils_g1_ggml[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            g1cur = vector_convnext_ggml(g1ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.6.convnext." + std::to_string(j),
                g1cur, dils_g1_ggml[j]);
            const std::string name = "ve_group1_convnext" + std::to_string(j);
            ggml_set_name(g1cur, name.c_str()); ggml_set_output(g1cur);
            ggml_build_forward_expand(g1gf, g1cur);
        }
        ggml_tensor * g1_tproj = ggml_mul_mat(g1ctx,
            ggml_cont(g1ctx, ggml_transpose(g1ctx, require_source_tensor(model, "vector_estimator:onnx::MatMul_3140"))),
            ggml_reshape_2d(g1ctx, g1_temb, 64, 1));
        g1_tproj = ggml_add(g1ctx, g1_tproj,
            ggml_reshape_2d(g1ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.7.linear.linear.bias"),
                C, 1));
        g1cur = ggml_add(g1ctx, g1cur, repeat_like(g1ctx, g1_tproj, g1cur));
        ggml_set_name(g1cur, "ve_group1_time_add"); ggml_set_output(g1cur);
        ggml_build_forward_expand(g1gf, g1cur);
        g1cur = vector_convnext_ggml(g1ctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.8.convnext.0",
            g1cur, 1);
        ggml_set_name(g1cur, "ve_group1_block8_convnext0"); ggml_set_output(g1cur);
        ggml_build_forward_expand(g1gf, g1cur);
        ggml_gallocr_t g1allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1allocr) throw std::runtime_error("ggml_gallocr_new group1 failed");
        if (!ggml_gallocr_reserve(g1allocr, g1gf)) {
            ggml_gallocr_free(g1allocr);
            throw std::runtime_error("ggml_gallocr_reserve group1 failed");
        }
        ggml_gallocr_alloc_graph(g1allocr, g1gf);
        std::vector<float> g1_in_raw = pack_time_channel_for_ggml(style_norm_ggml, L, C);
        ggml_backend_tensor_set(g1_in, g1_in_raw.data(), 0, g1_in_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_temb, te_host.data(), 0, te_host.size()*sizeof(float));
        supertonic_graph_compute(model, g1gf);
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_group1_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1gf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_group1_time_add", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1gf, "ve_group1_time_add"))});
        std::vector<float> g1_block8 = tensor_to_time_channel(ggml_graph_get_tensor(g1gf, "ve_group1_block8_convnext0"));
        PUSH_GGML_TRACE({"ve_group1_block8_convnext0", {L, C}, g1_block8});

        constexpr int G1_ATT_NODES = 512;
        static size_t g1_att_buf_size = ggml_tensor_overhead() * G1_ATT_NODES +
                                        ggml_graph_overhead_custom(G1_ATT_NODES, false);
        thread_local std::vector<uint8_t> g1_att_buf(g1_att_buf_size);
        ggml_init_params g1ap = { g1_att_buf_size, g1_att_buf.data(), true };
        ggml_context * g1actx = ggml_init(g1ap);
        ggml_cgraph * g1agf = ggml_new_graph_custom(g1actx, G1_ATT_NODES, false);
        ggml_tensor * g1a_x = ggml_new_tensor_2d(g1actx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1a_x, "g1a_x"); ggml_set_input(g1a_x);
        ggml_tensor * g1a_text = ggml_new_tensor_2d(g1actx, GGML_TYPE_F32, text_len, 256);
        ggml_set_name(g1a_text, "g1a_text"); ggml_set_input(g1a_text);
        ggml_tensor * g1a_q_rope = ggml_new_tensor_3d(g1actx, GGML_TYPE_F32, 64, L, 4);
        ggml_set_name(g1a_q_rope, "g1a_q_rope"); ggml_set_input(g1a_q_rope);
        ggml_tensor * g1a_k_rope = ggml_new_tensor_3d(g1actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g1a_k_rope, "g1a_k_rope"); ggml_set_input(g1a_k_rope);
        ggml_tensor * g1a_v_rope = ggml_new_tensor_3d(g1actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g1a_v_rope, "g1a_v_rope"); ggml_set_input(g1a_v_rope);
        ggml_tensor * g1q = dense_matmul_time_ggml(g1actx, g1a_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3146"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.W_query.linear.bias"));
        ggml_set_name(g1q, "ve_g1_attn_q"); ggml_set_output(g1q); ggml_build_forward_expand(g1agf, g1q);
        ggml_tensor * g1k = dense_matmul_time_ggml(g1actx, g1a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3147"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.W_key.linear.bias"));
        ggml_set_name(g1k, "ve_g1_attn_k"); ggml_set_output(g1k); ggml_build_forward_expand(g1agf, g1k);
        ggml_tensor * g1v = dense_matmul_time_ggml(g1actx, g1a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3148"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.W_value.linear.bias"));
        ggml_set_name(g1v, "ve_g1_attn_v"); ggml_set_output(g1v); ggml_build_forward_expand(g1agf, g1v);
        ggml_tensor * g1attn = ggml_flash_attn_ext(g1actx, g1a_q_rope, g1a_k_rope, g1a_v_rope, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g1attn = ggml_reshape_2d(g1actx, g1attn, 256, L);
        ggml_tensor * g1ctx_tc = ggml_cont(g1actx, ggml_transpose(g1actx, g1attn));
        ggml_set_name(g1ctx_tc, "ve_g1_attn_ctx"); ggml_set_output(g1ctx_tc); ggml_build_forward_expand(g1agf, g1ctx_tc);
        ggml_tensor * g1out = dense_matmul_time_ggml(g1actx, g1ctx_tc,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3155"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.attn.out_fc.linear.bias"));
        ggml_set_name(g1out, "ve_g1_attn_out"); ggml_set_output(g1out); ggml_build_forward_expand(g1agf, g1out);
        ggml_gallocr_t g1aallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1aallocr) throw std::runtime_error("ggml_gallocr_new group1 attn failed");
        if (!ggml_gallocr_reserve(g1aallocr, g1agf)) {
            ggml_gallocr_free(g1aallocr);
            throw std::runtime_error("ggml_gallocr_reserve group1 attn failed");
        }
        ggml_gallocr_alloc_graph(g1aallocr, g1agf);
        std::vector<float> g1a_x_raw = pack_time_channel_for_ggml(g1_block8, L, C);
        ggml_backend_tensor_set(g1a_x, g1a_x_raw.data(), 0, g1a_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1a_text, text_lc_host.data(), 0, text_lc_host.size()*sizeof(float));
        supertonic_graph_compute(model, g1agf);
        std::vector<float> g1q_out = tensor_to_time_channel(ggml_graph_get_tensor(g1agf, "ve_g1_attn_q"));
        std::vector<float> g1k_out = tensor_to_time_channel(ggml_graph_get_tensor(g1agf, "ve_g1_attn_k"));
        std::vector<float> g1v_out = tensor_to_time_channel(ggml_graph_get_tensor(g1agf, "ve_g1_attn_v"));
        PUSH_GGML_TRACE({"ve_g1_attn_q", {L, 256}, g1q_out});
        PUSH_GGML_TRACE({"ve_g1_attn_k", {text_len, 256}, g1k_out});
        PUSH_GGML_TRACE({"ve_g1_attn_v", {text_len, 256}, g1v_out});
        f32_tensor theta_g1 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g1.data.data(), g1q_out, L, 4, 64);
        apply_rope(theta_g1.data.data(), g1k_out, text_len, 4, 64);
        std::vector<float> g1q_dlh((size_t)64*L*4), g1k_dlh((size_t)64*text_len*4), g1v_dlh((size_t)64*text_len*4);
        for (int h = 0; h < 4; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 64; ++d) g1q_dlh[(size_t)d + 64*((size_t)t + (size_t)L*h)] = g1q_out[(size_t)t*256 + h*64 + d];
            for (int t = 0; t < text_len; ++t) for (int d = 0; d < 64; ++d) {
                g1k_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g1k_out[(size_t)t*256 + h*64 + d];
                g1v_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g1v_out[(size_t)t*256 + h*64 + d];
            }
        }
        ggml_backend_tensor_set(g1a_q_rope, g1q_dlh.data(), 0, g1q_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g1a_k_rope, g1k_dlh.data(), 0, g1k_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g1a_v_rope, g1v_dlh.data(), 0, g1v_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g1agf);
        PUSH_GGML_TRACE({"ve_g1_attn_q_rope", {L, 256}, g1q_out});
        PUSH_GGML_TRACE({"ve_g1_attn_k_rope", {text_len, 256}, g1k_out});
        PUSH_GGML_TRACE({"ve_g1_attn_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g1agf, "ve_g1_attn_ctx"))});
        std::vector<float> g1_attn_out = tensor_to_time_channel(ggml_graph_get_tensor(g1agf, "ve_g1_attn_out"));
        PUSH_GGML_TRACE({"ve_g1_attn_out", {L, C}, g1_attn_out});
        ggml_gallocr_free(g1aallocr);

        constexpr int G1_RES_NODES = 128;
        static size_t g1_res_buf_size = ggml_tensor_overhead() * G1_RES_NODES +
                                        ggml_graph_overhead_custom(G1_RES_NODES, false);
        thread_local std::vector<uint8_t> g1_res_buf(g1_res_buf_size);
        ggml_init_params g1rp = { g1_res_buf_size, g1_res_buf.data(), true };
        ggml_context * g1rctx = ggml_init(g1rp);
        ggml_cgraph * g1rgf = ggml_new_graph_custom(g1rctx, G1_RES_NODES, false);
        ggml_tensor * g1_lhs = ggml_new_tensor_2d(g1rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_lhs, "g1_res_lhs"); ggml_set_input(g1_lhs);
        ggml_tensor * g1_rhs = ggml_new_tensor_2d(g1rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_rhs, "g1_res_rhs"); ggml_set_input(g1_rhs);
        ggml_tensor * g1_res = ggml_add(g1rctx, g1_lhs, g1_rhs);
        ggml_set_name(g1_res, "ve_g1_attn_residual"); ggml_set_output(g1_res);
        ggml_build_forward_expand(g1rgf, g1_res);
        ggml_tensor * g1_norm = layer_norm_ggml(g1rctx, g1_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.9.norm.norm.bias"));
        ggml_set_name(g1_norm, "ve_g1_attn_norm"); ggml_set_output(g1_norm);
        ggml_build_forward_expand(g1rgf, g1_norm);
        ggml_tensor * g1_post = vector_convnext_ggml(g1rctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.10.convnext.0",
            g1_norm, 1);
        ggml_set_name(g1_post, "ve_g1_block10_convnext0"); ggml_set_output(g1_post);
        ggml_build_forward_expand(g1rgf, g1_post);
        ggml_gallocr_t g1rallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1rallocr) throw std::runtime_error("ggml_gallocr_new group1 residual failed");
        if (!ggml_gallocr_reserve(g1rallocr, g1rgf)) {
            ggml_gallocr_free(g1rallocr);
            throw std::runtime_error("ggml_gallocr_reserve group1 residual failed");
        }
        ggml_gallocr_alloc_graph(g1rallocr, g1rgf);
        std::vector<float> g1_lhs_raw = pack_time_channel_for_ggml(g1_block8, L, C);
        std::vector<float> g1_rhs_raw = pack_time_channel_for_ggml(g1_attn_out, L, C);
        ggml_backend_tensor_set(g1_lhs, g1_lhs_raw.data(), 0, g1_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_rhs, g1_rhs_raw.data(), 0, g1_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g1rgf);
        PUSH_GGML_TRACE({"ve_g1_attn_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1rgf, "ve_g1_attn_residual"))});
        PUSH_GGML_TRACE({"ve_g1_attn_norm", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1rgf, "ve_g1_attn_norm"))});
        std::vector<float> g1_block10 = tensor_to_time_channel(ggml_graph_get_tensor(g1rgf, "ve_g1_block10_convnext0"));
        PUSH_GGML_TRACE({"ve_g1_block10_convnext0", {L, C}, g1_block10});
        ggml_gallocr_free(g1rallocr);

        constexpr int G1_STYLE_NODES = 256;
        static size_t g1_style_buf_size = ggml_tensor_overhead() * G1_STYLE_NODES +
                                          ggml_graph_overhead_custom(G1_STYLE_NODES, false);
        thread_local std::vector<uint8_t> g1_style_buf(g1_style_buf_size);
        ggml_init_params g1sp = { g1_style_buf_size, g1_style_buf.data(), true };
        ggml_context * g1sctx = ggml_init(g1sp);
        ggml_cgraph * g1sgf = ggml_new_graph_custom(g1sctx, G1_STYLE_NODES, false);
        ggml_tensor * g1_style_x = ggml_new_tensor_2d(g1sctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_style_x, "g1_style_x"); ggml_set_input(g1_style_x);
        ggml_tensor * g1_style_v_in = ggml_new_tensor_2d(g1sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g1_style_v_in, "g1_style_ttl_lc"); ggml_set_input(g1_style_v_in);
        ggml_tensor * g1_kctx_in = ggml_new_tensor_2d(g1sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g1_kctx_in, "g1_style_kctx_lc"); ggml_set_input(g1_kctx_in);
        ggml_tensor * g1_style_q_dlh = ggml_new_tensor_3d(g1sctx, GGML_TYPE_F32, 128, L, 2);
        ggml_set_name(g1_style_q_dlh, "g1_style_q_dlh"); ggml_set_input(g1_style_q_dlh);
        ggml_tensor * g1_style_k_dlh = ggml_new_tensor_3d(g1sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g1_style_k_dlh, "g1_style_k_dlh"); ggml_set_input(g1_style_k_dlh);
        ggml_tensor * g1_style_v_dlh = ggml_new_tensor_3d(g1sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g1_style_v_dlh, "g1_style_v_dlh"); ggml_set_input(g1_style_v_dlh);
        ggml_tensor * g1sq = dense_matmul_time_ggml(g1sctx, g1_style_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3161"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_query.linear.bias"));
        ggml_tensor * g1sk = dense_matmul_time_ggml(g1sctx, g1_kctx_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3162"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_key.linear.bias"));
        g1sk = ggml_tanh(g1sctx, g1sk);
        ggml_tensor * g1sv = dense_matmul_time_ggml(g1sctx, g1_style_v_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3163"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.W_value.linear.bias"));
        ggml_set_name(g1sq, "ve_g1_style_q"); ggml_set_output(g1sq); ggml_build_forward_expand(g1sgf, g1sq);
        ggml_set_name(g1sk, "ve_g1_style_k_tanh"); ggml_set_output(g1sk); ggml_build_forward_expand(g1sgf, g1sk);
        ggml_set_name(g1sv, "ve_g1_style_v"); ggml_set_output(g1sv); ggml_build_forward_expand(g1sgf, g1sv);
        ggml_tensor * g1s_attn = ggml_flash_attn_ext(g1sctx, g1_style_q_dlh, g1_style_k_dlh, g1_style_v_dlh, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g1s_attn = ggml_reshape_2d(g1sctx, g1s_attn, 256, L);
        ggml_tensor * g1s_ctx = ggml_cont(g1sctx, ggml_transpose(g1sctx, g1s_attn));
        ggml_set_name(g1s_ctx, "ve_g1_style_ctx"); ggml_set_output(g1s_ctx); ggml_build_forward_expand(g1sgf, g1s_ctx);
        ggml_tensor * g1s_out = dense_matmul_time_ggml(g1sctx, g1s_ctx,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3164"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.attention.out_fc.linear.bias"));
        ggml_set_name(g1s_out, "ve_g1_style_out"); ggml_set_output(g1s_out); ggml_build_forward_expand(g1sgf, g1s_out);
        ggml_gallocr_t g1sallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1sallocr) throw std::runtime_error("ggml_gallocr_new group1 style failed");
        if (!ggml_gallocr_reserve(g1sallocr, g1sgf)) {
            ggml_gallocr_free(g1sallocr);
            throw std::runtime_error("ggml_gallocr_reserve group1 style failed");
        }
        ggml_gallocr_alloc_graph(g1sallocr, g1sgf);
        std::vector<float> g1_style_x_raw = pack_time_channel_for_ggml(g1_block10, L, C);
        ggml_backend_tensor_set(g1_style_x, g1_style_x_raw.data(), 0, g1_style_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_style_v_in, style_v_raw.data(), 0, style_v_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_kctx_in, kctx_raw.data(), 0, kctx_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g1sgf);
        std::vector<float> g1sq_out = tensor_to_time_channel(ggml_graph_get_tensor(g1sgf, "ve_g1_style_q"));
        std::vector<float> g1sk_out = tensor_to_time_channel(ggml_graph_get_tensor(g1sgf, "ve_g1_style_k_tanh"));
        std::vector<float> g1sv_out = tensor_to_time_channel(ggml_graph_get_tensor(g1sgf, "ve_g1_style_v"));
        PUSH_GGML_TRACE({"ve_g1_style_q", {L, 256}, g1sq_out});
        PUSH_GGML_TRACE({"ve_g1_style_k_tanh", {50, 256}, g1sk_out});
        PUSH_GGML_TRACE({"ve_g1_style_v", {50, 256}, g1sv_out});
        std::vector<float> g1sq_dlh((size_t)128*L*2), g1sk_dlh((size_t)128*50*2), g1sv_dlh((size_t)128*50*2);
        for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 128; ++d) g1sq_dlh[(size_t)d + 128*((size_t)t + (size_t)L*h)] = g1sq_out[(size_t)t*256 + h*128 + d];
            for (int t = 0; t < 50; ++t) for (int d = 0; d < 128; ++d) {
                g1sk_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g1sk_out[(size_t)t*256 + h*128 + d];
                g1sv_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g1sv_out[(size_t)t*256 + h*128 + d];
            }
        }
        ggml_backend_tensor_set(g1_style_q_dlh, g1sq_dlh.data(), 0, g1sq_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g1_style_k_dlh, g1sk_dlh.data(), 0, g1sk_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g1_style_v_dlh, g1sv_dlh.data(), 0, g1sv_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g1sgf);
        PUSH_GGML_TRACE({"ve_g1_style_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g1sgf, "ve_g1_style_ctx"))});
        std::vector<float> g1_style_out = tensor_to_time_channel(ggml_graph_get_tensor(g1sgf, "ve_g1_style_out"));
        PUSH_GGML_TRACE({"ve_g1_style_out", {L, C}, g1_style_out});
        ggml_gallocr_free(g1sallocr);

        constexpr int G1_STYLE_RES_NODES = 128;
        static size_t g1_style_res_buf_size = ggml_tensor_overhead() * G1_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G1_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g1_style_res_buf(g1_style_res_buf_size);
        ggml_init_params g1srp = { g1_style_res_buf_size, g1_style_res_buf.data(), true };
        ggml_context * g1srctx = ggml_init(g1srp);
        ggml_cgraph * g1srgf = ggml_new_graph_custom(g1srctx, G1_STYLE_RES_NODES, false);
        ggml_tensor * g1_style_lhs = ggml_new_tensor_2d(g1srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_style_lhs, "g1_style_lhs"); ggml_set_input(g1_style_lhs);
        ggml_tensor * g1_style_rhs = ggml_new_tensor_2d(g1srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g1_style_rhs, "g1_style_rhs"); ggml_set_input(g1_style_rhs);
        ggml_tensor * g1_style_res = ggml_add(g1srctx, g1_style_lhs, g1_style_rhs);
        ggml_set_name(g1_style_res, "ve_g1_style_residual"); ggml_set_output(g1_style_res);
        ggml_build_forward_expand(g1srgf, g1_style_res);
        ggml_tensor * g1_style_norm = layer_norm_ggml(g1srctx, g1_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.11.norm.norm.bias"));
        ggml_set_name(g1_style_norm, "ve_g1_style_norm"); ggml_set_output(g1_style_norm);
        ggml_build_forward_expand(g1srgf, g1_style_norm);
        ggml_gallocr_t g1srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g1srallocr) throw std::runtime_error("ggml_gallocr_new group1 style residual failed");
        if (!ggml_gallocr_reserve(g1srallocr, g1srgf)) {
            ggml_gallocr_free(g1srallocr);
            throw std::runtime_error("ggml_gallocr_reserve group1 style residual failed");
        }
        ggml_gallocr_alloc_graph(g1srallocr, g1srgf);
        std::vector<float> g1_style_lhs_raw = pack_time_channel_for_ggml(g1_block10, L, C);
        std::vector<float> g1_style_rhs_raw = pack_time_channel_for_ggml(g1_style_out, L, C);
        ggml_backend_tensor_set(g1_style_lhs, g1_style_lhs_raw.data(), 0, g1_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g1_style_rhs, g1_style_rhs_raw.data(), 0, g1_style_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g1srgf);
        PUSH_GGML_TRACE({"ve_g1_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g1srgf, "ve_g1_style_residual"))});
        std::vector<float> g1_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g1srgf, "ve_g1_style_norm"));
        PUSH_GGML_TRACE({"ve_g1_style_norm", {L, C}, g1_style_norm_vec});
        ggml_gallocr_free(g1srallocr);

        constexpr int G2_NODES = 512;
        static size_t g2_buf_size = ggml_tensor_overhead() * G2_NODES +
                                    ggml_graph_overhead_custom(G2_NODES, false);
        thread_local std::vector<uint8_t> g2_buf(g2_buf_size);
        ggml_init_params g2p = { g2_buf_size, g2_buf.data(), true };
        ggml_context * g2ctx = ggml_init(g2p);
        ggml_cgraph * g2gf = ggml_new_graph_custom(g2ctx, G2_NODES, false);
        ggml_tensor * g2_in = ggml_new_tensor_2d(g2ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_in, "g2_in"); ggml_set_input(g2_in);
        ggml_tensor * g2_temb = ggml_new_tensor_1d(g2ctx, GGML_TYPE_F32, 64);
        ggml_set_name(g2_temb, "g2_temb"); ggml_set_input(g2_temb);
        ggml_tensor * g2cur = g2_in;
        int dils_g2_ggml[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            g2cur = vector_convnext_ggml(g2ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.12.convnext." + std::to_string(j),
                g2cur, dils_g2_ggml[j]);
            const std::string name = "ve_group2_convnext" + std::to_string(j);
            ggml_set_name(g2cur, name.c_str()); ggml_set_output(g2cur);
            ggml_build_forward_expand(g2gf, g2cur);
        }
        ggml_tensor * g2_tproj = ggml_mul_mat(g2ctx,
            ggml_cont(g2ctx, ggml_transpose(g2ctx, require_source_tensor(model, "vector_estimator:onnx::MatMul_3185"))),
            ggml_reshape_2d(g2ctx, g2_temb, 64, 1));
        g2_tproj = ggml_add(g2ctx, g2_tproj,
            ggml_reshape_2d(g2ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.13.linear.linear.bias"),
                C, 1));
        g2cur = ggml_add(g2ctx, g2cur, repeat_like(g2ctx, g2_tproj, g2cur));
        ggml_set_name(g2cur, "ve_group2_time_add"); ggml_set_output(g2cur);
        ggml_build_forward_expand(g2gf, g2cur);
        g2cur = vector_convnext_ggml(g2ctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.14.convnext.0",
            g2cur, 1);
        ggml_set_name(g2cur, "ve_group2_block14_convnext0"); ggml_set_output(g2cur);
        ggml_build_forward_expand(g2gf, g2cur);
        ggml_gallocr_t g2allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2allocr) throw std::runtime_error("ggml_gallocr_new group2 failed");
        if (!ggml_gallocr_reserve(g2allocr, g2gf)) {
            ggml_gallocr_free(g2allocr);
            throw std::runtime_error("ggml_gallocr_reserve group2 failed");
        }
        ggml_gallocr_alloc_graph(g2allocr, g2gf);
        std::vector<float> g2_in_raw = pack_time_channel_for_ggml(g1_style_norm_vec, L, C);
        ggml_backend_tensor_set(g2_in, g2_in_raw.data(), 0, g2_in_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_temb, te_host.data(), 0, te_host.size()*sizeof(float));
        supertonic_graph_compute(model, g2gf);
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_group2_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2gf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_group2_time_add", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2gf, "ve_group2_time_add"))});
        std::vector<float> g2_block14 = tensor_to_time_channel(ggml_graph_get_tensor(g2gf, "ve_group2_block14_convnext0"));
        PUSH_GGML_TRACE({"ve_group2_block14_convnext0", {L, C}, g2_block14});

        constexpr int G2_ATT_NODES = 512;
        static size_t g2_att_buf_size = ggml_tensor_overhead() * G2_ATT_NODES +
                                        ggml_graph_overhead_custom(G2_ATT_NODES, false);
        thread_local std::vector<uint8_t> g2_att_buf(g2_att_buf_size);
        ggml_init_params g2ap = { g2_att_buf_size, g2_att_buf.data(), true };
        ggml_context * g2actx = ggml_init(g2ap);
        ggml_cgraph * g2agf = ggml_new_graph_custom(g2actx, G2_ATT_NODES, false);
        ggml_tensor * g2a_x = ggml_new_tensor_2d(g2actx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2a_x, "g2a_x"); ggml_set_input(g2a_x);
        ggml_tensor * g2a_text = ggml_new_tensor_2d(g2actx, GGML_TYPE_F32, text_len, 256);
        ggml_set_name(g2a_text, "g2a_text"); ggml_set_input(g2a_text);
        ggml_tensor * g2a_q_rope = ggml_new_tensor_3d(g2actx, GGML_TYPE_F32, 64, L, 4);
        ggml_set_name(g2a_q_rope, "g2a_q_rope"); ggml_set_input(g2a_q_rope);
        ggml_tensor * g2a_k_rope = ggml_new_tensor_3d(g2actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g2a_k_rope, "g2a_k_rope"); ggml_set_input(g2a_k_rope);
        ggml_tensor * g2a_v_rope = ggml_new_tensor_3d(g2actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g2a_v_rope, "g2a_v_rope"); ggml_set_input(g2a_v_rope);
        ggml_tensor * g2q = dense_matmul_time_ggml(g2actx, g2a_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3191"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.W_query.linear.bias"));
        ggml_set_name(g2q, "ve_g2_attn_q"); ggml_set_output(g2q); ggml_build_forward_expand(g2agf, g2q);
        ggml_tensor * g2k = dense_matmul_time_ggml(g2actx, g2a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3192"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.W_key.linear.bias"));
        ggml_set_name(g2k, "ve_g2_attn_k"); ggml_set_output(g2k); ggml_build_forward_expand(g2agf, g2k);
        ggml_tensor * g2v = dense_matmul_time_ggml(g2actx, g2a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3193"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.W_value.linear.bias"));
        ggml_set_name(g2v, "ve_g2_attn_v"); ggml_set_output(g2v); ggml_build_forward_expand(g2agf, g2v);
        ggml_tensor * g2attn = ggml_flash_attn_ext(g2actx, g2a_q_rope, g2a_k_rope, g2a_v_rope, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g2attn = ggml_reshape_2d(g2actx, g2attn, 256, L);
        ggml_tensor * g2ctx_tc = ggml_cont(g2actx, ggml_transpose(g2actx, g2attn));
        ggml_set_name(g2ctx_tc, "ve_g2_attn_ctx"); ggml_set_output(g2ctx_tc); ggml_build_forward_expand(g2agf, g2ctx_tc);
        ggml_tensor * g2out = dense_matmul_time_ggml(g2actx, g2ctx_tc,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3200"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.attn.out_fc.linear.bias"));
        ggml_set_name(g2out, "ve_g2_attn_out"); ggml_set_output(g2out); ggml_build_forward_expand(g2agf, g2out);
        ggml_gallocr_t g2aallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2aallocr) throw std::runtime_error("ggml_gallocr_new group2 attn failed");
        if (!ggml_gallocr_reserve(g2aallocr, g2agf)) {
            ggml_gallocr_free(g2aallocr);
            throw std::runtime_error("ggml_gallocr_reserve group2 attn failed");
        }
        ggml_gallocr_alloc_graph(g2aallocr, g2agf);
        std::vector<float> g2a_x_raw = pack_time_channel_for_ggml(g2_block14, L, C);
        ggml_backend_tensor_set(g2a_x, g2a_x_raw.data(), 0, g2a_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2a_text, text_lc_host.data(), 0, text_lc_host.size()*sizeof(float));
        supertonic_graph_compute(model, g2agf);
        std::vector<float> g2q_out = tensor_to_time_channel(ggml_graph_get_tensor(g2agf, "ve_g2_attn_q"));
        std::vector<float> g2k_out = tensor_to_time_channel(ggml_graph_get_tensor(g2agf, "ve_g2_attn_k"));
        std::vector<float> g2v_out = tensor_to_time_channel(ggml_graph_get_tensor(g2agf, "ve_g2_attn_v"));
        PUSH_GGML_TRACE({"ve_g2_attn_q", {L, 256}, g2q_out});
        PUSH_GGML_TRACE({"ve_g2_attn_k", {text_len, 256}, g2k_out});
        PUSH_GGML_TRACE({"ve_g2_attn_v", {text_len, 256}, g2v_out});
        f32_tensor theta_g2 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g2.data.data(), g2q_out, L, 4, 64);
        apply_rope(theta_g2.data.data(), g2k_out, text_len, 4, 64);
        std::vector<float> g2q_dlh((size_t)64*L*4), g2k_dlh((size_t)64*text_len*4), g2v_dlh((size_t)64*text_len*4);
        for (int h = 0; h < 4; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 64; ++d) g2q_dlh[(size_t)d + 64*((size_t)t + (size_t)L*h)] = g2q_out[(size_t)t*256 + h*64 + d];
            for (int t = 0; t < text_len; ++t) for (int d = 0; d < 64; ++d) {
                g2k_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g2k_out[(size_t)t*256 + h*64 + d];
                g2v_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g2v_out[(size_t)t*256 + h*64 + d];
            }
        }
        ggml_backend_tensor_set(g2a_q_rope, g2q_dlh.data(), 0, g2q_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g2a_k_rope, g2k_dlh.data(), 0, g2k_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g2a_v_rope, g2v_dlh.data(), 0, g2v_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g2agf);
        PUSH_GGML_TRACE({"ve_g2_attn_q_rope", {L, 256}, g2q_out});
        PUSH_GGML_TRACE({"ve_g2_attn_k_rope", {text_len, 256}, g2k_out});
        PUSH_GGML_TRACE({"ve_g2_attn_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g2agf, "ve_g2_attn_ctx"))});
        std::vector<float> g2_attn_out = tensor_to_time_channel(ggml_graph_get_tensor(g2agf, "ve_g2_attn_out"));
        PUSH_GGML_TRACE({"ve_g2_attn_out", {L, C}, g2_attn_out});
        ggml_gallocr_free(g2aallocr);

        constexpr int G2_RES_NODES = 128;
        static size_t g2_res_buf_size = ggml_tensor_overhead() * G2_RES_NODES +
                                        ggml_graph_overhead_custom(G2_RES_NODES, false);
        thread_local std::vector<uint8_t> g2_res_buf(g2_res_buf_size);
        ggml_init_params g2rp = { g2_res_buf_size, g2_res_buf.data(), true };
        ggml_context * g2rctx = ggml_init(g2rp);
        ggml_cgraph * g2rgf = ggml_new_graph_custom(g2rctx, G2_RES_NODES, false);
        ggml_tensor * g2_lhs = ggml_new_tensor_2d(g2rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_lhs, "g2_res_lhs"); ggml_set_input(g2_lhs);
        ggml_tensor * g2_rhs = ggml_new_tensor_2d(g2rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_rhs, "g2_res_rhs"); ggml_set_input(g2_rhs);
        ggml_tensor * g2_res = ggml_add(g2rctx, g2_lhs, g2_rhs);
        ggml_set_name(g2_res, "ve_g2_attn_residual"); ggml_set_output(g2_res);
        ggml_build_forward_expand(g2rgf, g2_res);
        ggml_tensor * g2_norm = layer_norm_ggml(g2rctx, g2_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.15.norm.norm.bias"));
        ggml_set_name(g2_norm, "ve_g2_attn_norm"); ggml_set_output(g2_norm);
        ggml_build_forward_expand(g2rgf, g2_norm);
        ggml_tensor * g2_post = vector_convnext_ggml(g2rctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.16.convnext.0",
            g2_norm, 1);
        ggml_set_name(g2_post, "ve_g2_block16_convnext0"); ggml_set_output(g2_post);
        ggml_build_forward_expand(g2rgf, g2_post);
        ggml_gallocr_t g2rallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2rallocr) throw std::runtime_error("ggml_gallocr_new group2 residual failed");
        if (!ggml_gallocr_reserve(g2rallocr, g2rgf)) {
            ggml_gallocr_free(g2rallocr);
            throw std::runtime_error("ggml_gallocr_reserve group2 residual failed");
        }
        ggml_gallocr_alloc_graph(g2rallocr, g2rgf);
        std::vector<float> g2_lhs_raw = pack_time_channel_for_ggml(g2_block14, L, C);
        std::vector<float> g2_rhs_raw = pack_time_channel_for_ggml(g2_attn_out, L, C);
        ggml_backend_tensor_set(g2_lhs, g2_lhs_raw.data(), 0, g2_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_rhs, g2_rhs_raw.data(), 0, g2_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g2rgf);
        PUSH_GGML_TRACE({"ve_g2_attn_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2rgf, "ve_g2_attn_residual"))});
        PUSH_GGML_TRACE({"ve_g2_attn_norm", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2rgf, "ve_g2_attn_norm"))});
        std::vector<float> g2_block16 = tensor_to_time_channel(ggml_graph_get_tensor(g2rgf, "ve_g2_block16_convnext0"));
        PUSH_GGML_TRACE({"ve_g2_block16_convnext0", {L, C}, g2_block16});
        ggml_gallocr_free(g2rallocr);

        constexpr int G2_STYLE_NODES = 256;
        static size_t g2_style_buf_size = ggml_tensor_overhead() * G2_STYLE_NODES +
                                          ggml_graph_overhead_custom(G2_STYLE_NODES, false);
        thread_local std::vector<uint8_t> g2_style_buf(g2_style_buf_size);
        ggml_init_params g2sp = { g2_style_buf_size, g2_style_buf.data(), true };
        ggml_context * g2sctx = ggml_init(g2sp);
        ggml_cgraph * g2sgf = ggml_new_graph_custom(g2sctx, G2_STYLE_NODES, false);
        ggml_tensor * g2_style_x = ggml_new_tensor_2d(g2sctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_style_x, "g2_style_x"); ggml_set_input(g2_style_x);
        ggml_tensor * g2_style_v_in = ggml_new_tensor_2d(g2sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g2_style_v_in, "g2_style_ttl_lc"); ggml_set_input(g2_style_v_in);
        ggml_tensor * g2_kctx_in = ggml_new_tensor_2d(g2sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g2_kctx_in, "g2_style_kctx_lc"); ggml_set_input(g2_kctx_in);
        ggml_tensor * g2_style_q_dlh = ggml_new_tensor_3d(g2sctx, GGML_TYPE_F32, 128, L, 2);
        ggml_set_name(g2_style_q_dlh, "g2_style_q_dlh"); ggml_set_input(g2_style_q_dlh);
        ggml_tensor * g2_style_k_dlh = ggml_new_tensor_3d(g2sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g2_style_k_dlh, "g2_style_k_dlh"); ggml_set_input(g2_style_k_dlh);
        ggml_tensor * g2_style_v_dlh = ggml_new_tensor_3d(g2sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g2_style_v_dlh, "g2_style_v_dlh"); ggml_set_input(g2_style_v_dlh);
        ggml_tensor * g2sq = dense_matmul_time_ggml(g2sctx, g2_style_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3206"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_query.linear.bias"));
        ggml_tensor * g2sk = dense_matmul_time_ggml(g2sctx, g2_kctx_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3207"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_key.linear.bias"));
        g2sk = ggml_tanh(g2sctx, g2sk);
        ggml_tensor * g2sv = dense_matmul_time_ggml(g2sctx, g2_style_v_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3208"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.W_value.linear.bias"));
        ggml_set_name(g2sq, "ve_g2_style_q"); ggml_set_output(g2sq); ggml_build_forward_expand(g2sgf, g2sq);
        ggml_set_name(g2sk, "ve_g2_style_k_tanh"); ggml_set_output(g2sk); ggml_build_forward_expand(g2sgf, g2sk);
        ggml_set_name(g2sv, "ve_g2_style_v"); ggml_set_output(g2sv); ggml_build_forward_expand(g2sgf, g2sv);
        ggml_tensor * g2s_attn = ggml_flash_attn_ext(g2sctx, g2_style_q_dlh, g2_style_k_dlh, g2_style_v_dlh, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g2s_attn = ggml_reshape_2d(g2sctx, g2s_attn, 256, L);
        ggml_tensor * g2s_ctx = ggml_cont(g2sctx, ggml_transpose(g2sctx, g2s_attn));
        ggml_set_name(g2s_ctx, "ve_g2_style_ctx"); ggml_set_output(g2s_ctx); ggml_build_forward_expand(g2sgf, g2s_ctx);
        ggml_tensor * g2s_out = dense_matmul_time_ggml(g2sctx, g2s_ctx,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3209"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.attention.out_fc.linear.bias"));
        ggml_set_name(g2s_out, "ve_g2_style_out"); ggml_set_output(g2s_out); ggml_build_forward_expand(g2sgf, g2s_out);
        ggml_gallocr_t g2sallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2sallocr) throw std::runtime_error("ggml_gallocr_new group2 style failed");
        if (!ggml_gallocr_reserve(g2sallocr, g2sgf)) {
            ggml_gallocr_free(g2sallocr);
            throw std::runtime_error("ggml_gallocr_reserve group2 style failed");
        }
        ggml_gallocr_alloc_graph(g2sallocr, g2sgf);
        std::vector<float> g2_style_x_raw = pack_time_channel_for_ggml(g2_block16, L, C);
        ggml_backend_tensor_set(g2_style_x, g2_style_x_raw.data(), 0, g2_style_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_style_v_in, style_v_raw.data(), 0, style_v_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_kctx_in, kctx_raw.data(), 0, kctx_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g2sgf);
        std::vector<float> g2sq_out = tensor_to_time_channel(ggml_graph_get_tensor(g2sgf, "ve_g2_style_q"));
        std::vector<float> g2sk_out = tensor_to_time_channel(ggml_graph_get_tensor(g2sgf, "ve_g2_style_k_tanh"));
        std::vector<float> g2sv_out = tensor_to_time_channel(ggml_graph_get_tensor(g2sgf, "ve_g2_style_v"));
        PUSH_GGML_TRACE({"ve_g2_style_q", {L, 256}, g2sq_out});
        PUSH_GGML_TRACE({"ve_g2_style_k_tanh", {50, 256}, g2sk_out});
        PUSH_GGML_TRACE({"ve_g2_style_v", {50, 256}, g2sv_out});
        std::vector<float> g2sq_dlh((size_t)128*L*2), g2sk_dlh((size_t)128*50*2), g2sv_dlh((size_t)128*50*2);
        for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 128; ++d) g2sq_dlh[(size_t)d + 128*((size_t)t + (size_t)L*h)] = g2sq_out[(size_t)t*256 + h*128 + d];
            for (int t = 0; t < 50; ++t) for (int d = 0; d < 128; ++d) {
                g2sk_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g2sk_out[(size_t)t*256 + h*128 + d];
                g2sv_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g2sv_out[(size_t)t*256 + h*128 + d];
            }
        }
        ggml_backend_tensor_set(g2_style_q_dlh, g2sq_dlh.data(), 0, g2sq_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g2_style_k_dlh, g2sk_dlh.data(), 0, g2sk_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g2_style_v_dlh, g2sv_dlh.data(), 0, g2sv_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g2sgf);
        PUSH_GGML_TRACE({"ve_g2_style_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g2sgf, "ve_g2_style_ctx"))});
        std::vector<float> g2_style_out = tensor_to_time_channel(ggml_graph_get_tensor(g2sgf, "ve_g2_style_out"));
        PUSH_GGML_TRACE({"ve_g2_style_out", {L, C}, g2_style_out});
        ggml_gallocr_free(g2sallocr);

        constexpr int G2_STYLE_RES_NODES = 128;
        static size_t g2_style_res_buf_size = ggml_tensor_overhead() * G2_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G2_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g2_style_res_buf(g2_style_res_buf_size);
        ggml_init_params g2srp = { g2_style_res_buf_size, g2_style_res_buf.data(), true };
        ggml_context * g2srctx = ggml_init(g2srp);
        ggml_cgraph * g2srgf = ggml_new_graph_custom(g2srctx, G2_STYLE_RES_NODES, false);
        ggml_tensor * g2_style_lhs = ggml_new_tensor_2d(g2srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_style_lhs, "g2_style_lhs"); ggml_set_input(g2_style_lhs);
        ggml_tensor * g2_style_rhs = ggml_new_tensor_2d(g2srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g2_style_rhs, "g2_style_rhs"); ggml_set_input(g2_style_rhs);
        ggml_tensor * g2_style_res = ggml_add(g2srctx, g2_style_lhs, g2_style_rhs);
        ggml_set_name(g2_style_res, "ve_g2_style_residual"); ggml_set_output(g2_style_res);
        ggml_build_forward_expand(g2srgf, g2_style_res);
        ggml_tensor * g2_style_norm = layer_norm_ggml(g2srctx, g2_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.17.norm.norm.bias"));
        ggml_set_name(g2_style_norm, "ve_g2_style_norm"); ggml_set_output(g2_style_norm);
        ggml_build_forward_expand(g2srgf, g2_style_norm);
        ggml_gallocr_t g2srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g2srallocr) throw std::runtime_error("ggml_gallocr_new group2 style residual failed");
        if (!ggml_gallocr_reserve(g2srallocr, g2srgf)) {
            ggml_gallocr_free(g2srallocr);
            throw std::runtime_error("ggml_gallocr_reserve group2 style residual failed");
        }
        ggml_gallocr_alloc_graph(g2srallocr, g2srgf);
        std::vector<float> g2_style_lhs_raw = pack_time_channel_for_ggml(g2_block16, L, C);
        std::vector<float> g2_style_rhs_raw = pack_time_channel_for_ggml(g2_style_out, L, C);
        ggml_backend_tensor_set(g2_style_lhs, g2_style_lhs_raw.data(), 0, g2_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g2_style_rhs, g2_style_rhs_raw.data(), 0, g2_style_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g2srgf);
        PUSH_GGML_TRACE({"ve_g2_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g2srgf, "ve_g2_style_residual"))});
        std::vector<float> g2_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g2srgf, "ve_g2_style_norm"));
        PUSH_GGML_TRACE({"ve_g2_style_norm", {L, C}, g2_style_norm_vec});
        ggml_gallocr_free(g2srallocr);

        constexpr int G3_NODES = 512;
        static size_t g3_buf_size = ggml_tensor_overhead() * G3_NODES +
                                    ggml_graph_overhead_custom(G3_NODES, false);
        thread_local std::vector<uint8_t> g3_buf(g3_buf_size);
        ggml_init_params g3p = { g3_buf_size, g3_buf.data(), true };
        ggml_context * g3ctx = ggml_init(g3p);
        ggml_cgraph * g3gf = ggml_new_graph_custom(g3ctx, G3_NODES, false);
        ggml_tensor * g3_in = ggml_new_tensor_2d(g3ctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_in, "g3_in"); ggml_set_input(g3_in);
        ggml_tensor * g3_temb = ggml_new_tensor_1d(g3ctx, GGML_TYPE_F32, 64);
        ggml_set_name(g3_temb, "g3_temb"); ggml_set_input(g3_temb);
        ggml_tensor * g3cur = g3_in;
        int dils_g3_ggml[4] = {1, 2, 4, 8};
        for (int j = 0; j < 4; ++j) {
            g3cur = vector_convnext_ggml(g3ctx, model,
                "vector_estimator:tts.ttl.vector_field.main_blocks.18.convnext." + std::to_string(j),
                g3cur, dils_g3_ggml[j]);
            const std::string name = "ve_group3_convnext" + std::to_string(j);
            ggml_set_name(g3cur, name.c_str()); ggml_set_output(g3cur);
            ggml_build_forward_expand(g3gf, g3cur);
        }
        ggml_tensor * g3_tproj = ggml_mul_mat(g3ctx,
            ggml_cont(g3ctx, ggml_transpose(g3ctx, require_source_tensor(model, "vector_estimator:onnx::MatMul_3230"))),
            ggml_reshape_2d(g3ctx, g3_temb, 64, 1));
        g3_tproj = ggml_add(g3ctx, g3_tproj,
            ggml_reshape_2d(g3ctx,
                require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.19.linear.linear.bias"),
                C, 1));
        g3cur = ggml_add(g3ctx, g3cur, repeat_like(g3ctx, g3_tproj, g3cur));
        ggml_set_name(g3cur, "ve_group3_time_add"); ggml_set_output(g3cur);
        ggml_build_forward_expand(g3gf, g3cur);
        g3cur = vector_convnext_ggml(g3ctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.20.convnext.0",
            g3cur, 1);
        ggml_set_name(g3cur, "ve_group3_block20_convnext0"); ggml_set_output(g3cur);
        ggml_build_forward_expand(g3gf, g3cur);
        ggml_gallocr_t g3allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3allocr) throw std::runtime_error("ggml_gallocr_new group3 failed");
        if (!ggml_gallocr_reserve(g3allocr, g3gf)) {
            ggml_gallocr_free(g3allocr);
            throw std::runtime_error("ggml_gallocr_reserve group3 failed");
        }
        ggml_gallocr_alloc_graph(g3allocr, g3gf);
        std::vector<float> g3_in_raw = pack_time_channel_for_ggml(g2_style_norm_vec, L, C);
        ggml_backend_tensor_set(g3_in, g3_in_raw.data(), 0, g3_in_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_temb, te_host.data(), 0, te_host.size()*sizeof(float));
        supertonic_graph_compute(model, g3gf);
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_group3_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3gf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_group3_time_add", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3gf, "ve_group3_time_add"))});
        std::vector<float> g3_block20 = tensor_to_time_channel(ggml_graph_get_tensor(g3gf, "ve_group3_block20_convnext0"));
        PUSH_GGML_TRACE({"ve_group3_block20_convnext0", {L, C}, g3_block20});

        constexpr int G3_ATT_NODES = 512;
        static size_t g3_att_buf_size = ggml_tensor_overhead() * G3_ATT_NODES +
                                        ggml_graph_overhead_custom(G3_ATT_NODES, false);
        thread_local std::vector<uint8_t> g3_att_buf(g3_att_buf_size);
        ggml_init_params g3ap = { g3_att_buf_size, g3_att_buf.data(), true };
        ggml_context * g3actx = ggml_init(g3ap);
        ggml_cgraph * g3agf = ggml_new_graph_custom(g3actx, G3_ATT_NODES, false);
        ggml_tensor * g3a_x = ggml_new_tensor_2d(g3actx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3a_x, "g3a_x"); ggml_set_input(g3a_x);
        ggml_tensor * g3a_text = ggml_new_tensor_2d(g3actx, GGML_TYPE_F32, text_len, 256);
        ggml_set_name(g3a_text, "g3a_text"); ggml_set_input(g3a_text);
        ggml_tensor * g3a_q_rope = ggml_new_tensor_3d(g3actx, GGML_TYPE_F32, 64, L, 4);
        ggml_set_name(g3a_q_rope, "g3a_q_rope"); ggml_set_input(g3a_q_rope);
        ggml_tensor * g3a_k_rope = ggml_new_tensor_3d(g3actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g3a_k_rope, "g3a_k_rope"); ggml_set_input(g3a_k_rope);
        ggml_tensor * g3a_v_rope = ggml_new_tensor_3d(g3actx, GGML_TYPE_F32, 64, text_len, 4);
        ggml_set_name(g3a_v_rope, "g3a_v_rope"); ggml_set_input(g3a_v_rope);
        ggml_tensor * g3q = dense_matmul_time_ggml(g3actx, g3a_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3236"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.W_query.linear.bias"));
        ggml_set_name(g3q, "ve_g3_attn_q"); ggml_set_output(g3q); ggml_build_forward_expand(g3agf, g3q);
        ggml_tensor * g3k = dense_matmul_time_ggml(g3actx, g3a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3237"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.W_key.linear.bias"));
        ggml_set_name(g3k, "ve_g3_attn_k"); ggml_set_output(g3k); ggml_build_forward_expand(g3agf, g3k);
        ggml_tensor * g3v = dense_matmul_time_ggml(g3actx, g3a_text,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3238"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.W_value.linear.bias"));
        ggml_set_name(g3v, "ve_g3_attn_v"); ggml_set_output(g3v); ggml_build_forward_expand(g3agf, g3v);
        ggml_tensor * g3attn = ggml_flash_attn_ext(g3actx, g3a_q_rope, g3a_k_rope, g3a_v_rope, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g3attn = ggml_reshape_2d(g3actx, g3attn, 256, L);
        ggml_tensor * g3ctx_tc = ggml_cont(g3actx, ggml_transpose(g3actx, g3attn));
        ggml_set_name(g3ctx_tc, "ve_g3_attn_ctx"); ggml_set_output(g3ctx_tc); ggml_build_forward_expand(g3agf, g3ctx_tc);
        ggml_tensor * g3out = dense_matmul_time_ggml(g3actx, g3ctx_tc,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3245"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.attn.out_fc.linear.bias"));
        ggml_set_name(g3out, "ve_g3_attn_out"); ggml_set_output(g3out); ggml_build_forward_expand(g3agf, g3out);
        ggml_gallocr_t g3aallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3aallocr) throw std::runtime_error("ggml_gallocr_new group3 attn failed");
        if (!ggml_gallocr_reserve(g3aallocr, g3agf)) {
            ggml_gallocr_free(g3aallocr);
            throw std::runtime_error("ggml_gallocr_reserve group3 attn failed");
        }
        ggml_gallocr_alloc_graph(g3aallocr, g3agf);
        std::vector<float> g3a_x_raw = pack_time_channel_for_ggml(g3_block20, L, C);
        ggml_backend_tensor_set(g3a_x, g3a_x_raw.data(), 0, g3a_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3a_text, text_lc_host.data(), 0, text_lc_host.size()*sizeof(float));
        supertonic_graph_compute(model, g3agf);
        std::vector<float> g3q_out = tensor_to_time_channel(ggml_graph_get_tensor(g3agf, "ve_g3_attn_q"));
        std::vector<float> g3k_out = tensor_to_time_channel(ggml_graph_get_tensor(g3agf, "ve_g3_attn_k"));
        std::vector<float> g3v_out = tensor_to_time_channel(ggml_graph_get_tensor(g3agf, "ve_g3_attn_v"));
        PUSH_GGML_TRACE({"ve_g3_attn_q", {L, 256}, g3q_out});
        PUSH_GGML_TRACE({"ve_g3_attn_k", {text_len, 256}, g3k_out});
        PUSH_GGML_TRACE({"ve_g3_attn_v", {text_len, 256}, g3v_out});
        f32_tensor theta_g3 = read_f32(model, "vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta");
        apply_rope(theta_g3.data.data(), g3q_out, L, 4, 64);
        apply_rope(theta_g3.data.data(), g3k_out, text_len, 4, 64);
        std::vector<float> g3q_dlh((size_t)64*L*4), g3k_dlh((size_t)64*text_len*4), g3v_dlh((size_t)64*text_len*4);
        for (int h = 0; h < 4; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 64; ++d) g3q_dlh[(size_t)d + 64*((size_t)t + (size_t)L*h)] = g3q_out[(size_t)t*256 + h*64 + d];
            for (int t = 0; t < text_len; ++t) for (int d = 0; d < 64; ++d) {
                g3k_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g3k_out[(size_t)t*256 + h*64 + d];
                g3v_dlh[(size_t)d + 64*((size_t)t + (size_t)text_len*h)] = g3v_out[(size_t)t*256 + h*64 + d];
            }
        }
        ggml_backend_tensor_set(g3a_q_rope, g3q_dlh.data(), 0, g3q_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g3a_k_rope, g3k_dlh.data(), 0, g3k_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g3a_v_rope, g3v_dlh.data(), 0, g3v_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g3agf);
        PUSH_GGML_TRACE({"ve_g3_attn_q_rope", {L, 256}, g3q_out});
        PUSH_GGML_TRACE({"ve_g3_attn_k_rope", {text_len, 256}, g3k_out});
        PUSH_GGML_TRACE({"ve_g3_attn_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g3agf, "ve_g3_attn_ctx"))});
        std::vector<float> g3_attn_out = tensor_to_time_channel(ggml_graph_get_tensor(g3agf, "ve_g3_attn_out"));
        PUSH_GGML_TRACE({"ve_g3_attn_out", {L, C}, g3_attn_out});
        ggml_gallocr_free(g3aallocr);

        constexpr int G3_RES_NODES = 128;
        static size_t g3_res_buf_size = ggml_tensor_overhead() * G3_RES_NODES +
                                        ggml_graph_overhead_custom(G3_RES_NODES, false);
        thread_local std::vector<uint8_t> g3_res_buf(g3_res_buf_size);
        ggml_init_params g3rp = { g3_res_buf_size, g3_res_buf.data(), true };
        ggml_context * g3rctx = ggml_init(g3rp);
        ggml_cgraph * g3rgf = ggml_new_graph_custom(g3rctx, G3_RES_NODES, false);
        ggml_tensor * g3_lhs = ggml_new_tensor_2d(g3rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_lhs, "g3_res_lhs"); ggml_set_input(g3_lhs);
        ggml_tensor * g3_rhs = ggml_new_tensor_2d(g3rctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_rhs, "g3_res_rhs"); ggml_set_input(g3_rhs);
        ggml_tensor * g3_res = ggml_add(g3rctx, g3_lhs, g3_rhs);
        ggml_set_name(g3_res, "ve_g3_attn_residual"); ggml_set_output(g3_res);
        ggml_build_forward_expand(g3rgf, g3_res);
        ggml_tensor * g3_norm = layer_norm_ggml(g3rctx, g3_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.21.norm.norm.bias"));
        ggml_set_name(g3_norm, "ve_g3_attn_norm"); ggml_set_output(g3_norm);
        ggml_build_forward_expand(g3rgf, g3_norm);
        ggml_tensor * g3_post = vector_convnext_ggml(g3rctx, model,
            "vector_estimator:tts.ttl.vector_field.main_blocks.22.convnext.0",
            g3_norm, 1);
        ggml_set_name(g3_post, "ve_g3_block22_convnext0"); ggml_set_output(g3_post);
        ggml_build_forward_expand(g3rgf, g3_post);
        ggml_gallocr_t g3rallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3rallocr) throw std::runtime_error("ggml_gallocr_new group3 residual failed");
        if (!ggml_gallocr_reserve(g3rallocr, g3rgf)) {
            ggml_gallocr_free(g3rallocr);
            throw std::runtime_error("ggml_gallocr_reserve group3 residual failed");
        }
        ggml_gallocr_alloc_graph(g3rallocr, g3rgf);
        std::vector<float> g3_lhs_raw = pack_time_channel_for_ggml(g3_block20, L, C);
        std::vector<float> g3_rhs_raw = pack_time_channel_for_ggml(g3_attn_out, L, C);
        ggml_backend_tensor_set(g3_lhs, g3_lhs_raw.data(), 0, g3_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_rhs, g3_rhs_raw.data(), 0, g3_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g3rgf);
        PUSH_GGML_TRACE({"ve_g3_attn_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3rgf, "ve_g3_attn_residual"))});
        PUSH_GGML_TRACE({"ve_g3_attn_norm", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3rgf, "ve_g3_attn_norm"))});
        std::vector<float> g3_block22 = tensor_to_time_channel(ggml_graph_get_tensor(g3rgf, "ve_g3_block22_convnext0"));
        PUSH_GGML_TRACE({"ve_g3_block22_convnext0", {L, C}, g3_block22});
        ggml_gallocr_free(g3rallocr);

        constexpr int G3_STYLE_NODES = 256;
        static size_t g3_style_buf_size = ggml_tensor_overhead() * G3_STYLE_NODES +
                                          ggml_graph_overhead_custom(G3_STYLE_NODES, false);
        thread_local std::vector<uint8_t> g3_style_buf(g3_style_buf_size);
        ggml_init_params g3sp = { g3_style_buf_size, g3_style_buf.data(), true };
        ggml_context * g3sctx = ggml_init(g3sp);
        ggml_cgraph * g3sgf = ggml_new_graph_custom(g3sctx, G3_STYLE_NODES, false);
        ggml_tensor * g3_style_x = ggml_new_tensor_2d(g3sctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_style_x, "g3_style_x"); ggml_set_input(g3_style_x);
        ggml_tensor * g3_style_v_in = ggml_new_tensor_2d(g3sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g3_style_v_in, "g3_style_ttl_lc"); ggml_set_input(g3_style_v_in);
        ggml_tensor * g3_kctx_in = ggml_new_tensor_2d(g3sctx, GGML_TYPE_F32, 50, 256);
        ggml_set_name(g3_kctx_in, "g3_style_kctx_lc"); ggml_set_input(g3_kctx_in);
        ggml_tensor * g3_style_q_dlh = ggml_new_tensor_3d(g3sctx, GGML_TYPE_F32, 128, L, 2);
        ggml_set_name(g3_style_q_dlh, "g3_style_q_dlh"); ggml_set_input(g3_style_q_dlh);
        ggml_tensor * g3_style_k_dlh = ggml_new_tensor_3d(g3sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g3_style_k_dlh, "g3_style_k_dlh"); ggml_set_input(g3_style_k_dlh);
        ggml_tensor * g3_style_v_dlh = ggml_new_tensor_3d(g3sctx, GGML_TYPE_F32, 128, 50, 2);
        ggml_set_name(g3_style_v_dlh, "g3_style_v_dlh"); ggml_set_input(g3_style_v_dlh);
        ggml_tensor * g3sq = dense_matmul_time_ggml(g3sctx, g3_style_x,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3251"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_query.linear.bias"));
        ggml_tensor * g3sk = dense_matmul_time_ggml(g3sctx, g3_kctx_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3252"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_key.linear.bias"));
        g3sk = ggml_tanh(g3sctx, g3sk);
        ggml_tensor * g3sv = dense_matmul_time_ggml(g3sctx, g3_style_v_in,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3253"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.W_value.linear.bias"));
        ggml_set_name(g3sq, "ve_g3_style_q"); ggml_set_output(g3sq); ggml_build_forward_expand(g3sgf, g3sq);
        ggml_set_name(g3sk, "ve_g3_style_k_tanh"); ggml_set_output(g3sk); ggml_build_forward_expand(g3sgf, g3sk);
        ggml_set_name(g3sv, "ve_g3_style_v"); ggml_set_output(g3sv); ggml_build_forward_expand(g3sgf, g3sv);
        ggml_tensor * g3s_attn = ggml_flash_attn_ext(g3sctx, g3_style_q_dlh, g3_style_k_dlh, g3_style_v_dlh, nullptr, 1.0f/16.0f, 0.0f, 0.0f);
        g3s_attn = ggml_reshape_2d(g3sctx, g3s_attn, 256, L);
        ggml_tensor * g3s_ctx = ggml_cont(g3sctx, ggml_transpose(g3sctx, g3s_attn));
        ggml_set_name(g3s_ctx, "ve_g3_style_ctx"); ggml_set_output(g3s_ctx); ggml_build_forward_expand(g3sgf, g3s_ctx);
        ggml_tensor * g3s_out = dense_matmul_time_ggml(g3sctx, g3s_ctx,
            require_source_tensor(model, "vector_estimator:onnx::MatMul_3254"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.attention.out_fc.linear.bias"));
        ggml_set_name(g3s_out, "ve_g3_style_out"); ggml_set_output(g3s_out); ggml_build_forward_expand(g3sgf, g3s_out);
        ggml_gallocr_t g3sallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3sallocr) throw std::runtime_error("ggml_gallocr_new group3 style failed");
        if (!ggml_gallocr_reserve(g3sallocr, g3sgf)) {
            ggml_gallocr_free(g3sallocr);
            throw std::runtime_error("ggml_gallocr_reserve group3 style failed");
        }
        ggml_gallocr_alloc_graph(g3sallocr, g3sgf);
        std::vector<float> g3_style_x_raw = pack_time_channel_for_ggml(g3_block22, L, C);
        ggml_backend_tensor_set(g3_style_x, g3_style_x_raw.data(), 0, g3_style_x_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_style_v_in, style_v_raw.data(), 0, style_v_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_kctx_in, kctx_raw.data(), 0, kctx_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g3sgf);
        std::vector<float> g3sq_out = tensor_to_time_channel(ggml_graph_get_tensor(g3sgf, "ve_g3_style_q"));
        std::vector<float> g3sk_out = tensor_to_time_channel(ggml_graph_get_tensor(g3sgf, "ve_g3_style_k_tanh"));
        std::vector<float> g3sv_out = tensor_to_time_channel(ggml_graph_get_tensor(g3sgf, "ve_g3_style_v"));
        PUSH_GGML_TRACE({"ve_g3_style_q", {L, 256}, g3sq_out});
        PUSH_GGML_TRACE({"ve_g3_style_k_tanh", {50, 256}, g3sk_out});
        PUSH_GGML_TRACE({"ve_g3_style_v", {50, 256}, g3sv_out});
        std::vector<float> g3sq_dlh((size_t)128*L*2), g3sk_dlh((size_t)128*50*2), g3sv_dlh((size_t)128*50*2);
        for (int h = 0; h < 2; ++h) {
            for (int t = 0; t < L; ++t) for (int d = 0; d < 128; ++d) g3sq_dlh[(size_t)d + 128*((size_t)t + (size_t)L*h)] = g3sq_out[(size_t)t*256 + h*128 + d];
            for (int t = 0; t < 50; ++t) for (int d = 0; d < 128; ++d) {
                g3sk_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g3sk_out[(size_t)t*256 + h*128 + d];
                g3sv_dlh[(size_t)d + 128*((size_t)t + 50ULL*h)] = g3sv_out[(size_t)t*256 + h*128 + d];
            }
        }
        ggml_backend_tensor_set(g3_style_q_dlh, g3sq_dlh.data(), 0, g3sq_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g3_style_k_dlh, g3sk_dlh.data(), 0, g3sk_dlh.size()*sizeof(float));
        ggml_backend_tensor_set(g3_style_v_dlh, g3sv_dlh.data(), 0, g3sv_dlh.size()*sizeof(float));
        supertonic_graph_compute(model, g3sgf);
        PUSH_GGML_TRACE({"ve_g3_style_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(g3sgf, "ve_g3_style_ctx"))});
        std::vector<float> g3_style_out = tensor_to_time_channel(ggml_graph_get_tensor(g3sgf, "ve_g3_style_out"));
        PUSH_GGML_TRACE({"ve_g3_style_out", {L, C}, g3_style_out});
        ggml_gallocr_free(g3sallocr);

        constexpr int G3_STYLE_RES_NODES = 128;
        static size_t g3_style_res_buf_size = ggml_tensor_overhead() * G3_STYLE_RES_NODES +
                                              ggml_graph_overhead_custom(G3_STYLE_RES_NODES, false);
        thread_local std::vector<uint8_t> g3_style_res_buf(g3_style_res_buf_size);
        ggml_init_params g3srp = { g3_style_res_buf_size, g3_style_res_buf.data(), true };
        ggml_context * g3srctx = ggml_init(g3srp);
        ggml_cgraph * g3srgf = ggml_new_graph_custom(g3srctx, G3_STYLE_RES_NODES, false);
        ggml_tensor * g3_style_lhs = ggml_new_tensor_2d(g3srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_style_lhs, "g3_style_lhs"); ggml_set_input(g3_style_lhs);
        ggml_tensor * g3_style_rhs = ggml_new_tensor_2d(g3srctx, GGML_TYPE_F32, L, C);
        ggml_set_name(g3_style_rhs, "g3_style_rhs"); ggml_set_input(g3_style_rhs);
        ggml_tensor * g3_style_res = ggml_add(g3srctx, g3_style_lhs, g3_style_rhs);
        ggml_set_name(g3_style_res, "ve_g3_style_residual"); ggml_set_output(g3_style_res);
        ggml_build_forward_expand(g3srgf, g3_style_res);
        ggml_tensor * g3_style_norm = layer_norm_ggml(g3srctx, g3_style_res,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.weight"),
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.main_blocks.23.norm.norm.bias"));
        ggml_set_name(g3_style_norm, "ve_g3_style_norm"); ggml_set_output(g3_style_norm);
        ggml_build_forward_expand(g3srgf, g3_style_norm);
        ggml_gallocr_t g3srallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!g3srallocr) throw std::runtime_error("ggml_gallocr_new group3 style residual failed");
        if (!ggml_gallocr_reserve(g3srallocr, g3srgf)) {
            ggml_gallocr_free(g3srallocr);
            throw std::runtime_error("ggml_gallocr_reserve group3 style residual failed");
        }
        ggml_gallocr_alloc_graph(g3srallocr, g3srgf);
        std::vector<float> g3_style_lhs_raw = pack_time_channel_for_ggml(g3_block22, L, C);
        std::vector<float> g3_style_rhs_raw = pack_time_channel_for_ggml(g3_style_out, L, C);
        ggml_backend_tensor_set(g3_style_lhs, g3_style_lhs_raw.data(), 0, g3_style_lhs_raw.size()*sizeof(float));
        ggml_backend_tensor_set(g3_style_rhs, g3_style_rhs_raw.data(), 0, g3_style_rhs_raw.size()*sizeof(float));
        supertonic_graph_compute(model, g3srgf);
        PUSH_GGML_TRACE({"ve_g3_style_residual", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(g3srgf, "ve_g3_style_residual"))});
        std::vector<float> g3_style_norm_vec = tensor_to_time_channel(ggml_graph_get_tensor(g3srgf, "ve_g3_style_norm"));
        PUSH_GGML_TRACE({"ve_g3_style_norm", {L, C}, g3_style_norm_vec});
        ggml_gallocr_free(g3srallocr);

        constexpr int TAIL_NODES = 512;
        static size_t tail_buf_size = ggml_tensor_overhead() * TAIL_NODES +
                                      ggml_graph_overhead_custom(TAIL_NODES, false);
        thread_local std::vector<uint8_t> tail_buf(tail_buf_size);
        ggml_init_params tailp = { tail_buf_size, tail_buf.data(), true };
        ggml_context * tailctx = ggml_init(tailp);
        ggml_cgraph * tailgf = ggml_new_graph_custom(tailctx, TAIL_NODES, false);
        ggml_tensor * tail_in = ggml_new_tensor_2d(tailctx, GGML_TYPE_F32, L, C);
        ggml_set_name(tail_in, "tail_in"); ggml_set_input(tail_in);
        ggml_tensor * tail_mask = ggml_new_tensor_1d(tailctx, GGML_TYPE_F32, L);
        ggml_set_name(tail_mask, "tail_mask"); ggml_set_input(tail_mask);
        ggml_tensor * tail_noise = ggml_new_tensor_2d(tailctx, GGML_TYPE_F32, L, Cin);
        ggml_set_name(tail_noise, "tail_noise"); ggml_set_input(tail_noise);
        ggml_tensor * tail = tail_in;
        for (int j = 0; j < 4; ++j) {
            tail = vector_convnext_ggml(tailctx, model,
                "vector_estimator:tts.ttl.vector_field.last_convnext.convnext." + std::to_string(j),
                tail, 1);
            const std::string name = "ve_last_convnext" + std::to_string(j);
            ggml_set_name(tail, name.c_str()); ggml_set_output(tail);
            ggml_build_forward_expand(tailgf, tail);
        }
        ggml_tensor * velocity_t = conv1d_f32(tailctx,
            require_source_tensor(model, "vector_estimator:tts.ttl.vector_field.proj_out.net.weight"),
            tail, 1, 0, 1);
        velocity_t = ggml_mul(tailctx, velocity_t, repeat_like(tailctx, tail_mask, velocity_t));
        ggml_set_name(velocity_t, "ve_proj_out"); ggml_set_output(velocity_t);
        ggml_build_forward_expand(tailgf, velocity_t);
        ggml_tensor * next = ggml_add(tailctx, tail_noise, ggml_scale(tailctx, velocity_t, 1.0f/(float) total_steps));
        ggml_set_name(next, "ve_next_latent_tc"); ggml_set_output(next);
        ggml_build_forward_expand(tailgf, next);
        ggml_gallocr_t tailallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        if (!tailallocr) throw std::runtime_error("ggml_gallocr_new tail failed");
        if (!ggml_gallocr_reserve(tailallocr, tailgf)) {
            ggml_gallocr_free(tailallocr);
            throw std::runtime_error("ggml_gallocr_reserve tail failed");
        }
        ggml_gallocr_alloc_graph(tailallocr, tailgf);
        std::vector<float> tail_in_raw = pack_time_channel_for_ggml(g3_style_norm_vec, L, C);
        std::vector<float> noise_tc((size_t)L*Cin);
        for (int t = 0; t < L; ++t) for (int c = 0; c < Cin; ++c) noise_tc[(size_t)t*Cin+c] = noisy_latent[(size_t)c*L+t];
        std::vector<float> noise_raw = pack_time_channel_for_ggml(noise_tc, L, Cin);
        ggml_backend_tensor_set(tail_in, tail_in_raw.data(), 0, tail_in_raw.size()*sizeof(float));
        ggml_backend_tensor_set(tail_mask, latent_mask, 0, (size_t)L*sizeof(float));
        ggml_backend_tensor_set(tail_noise, noise_raw.data(), 0, noise_raw.size()*sizeof(float));
        supertonic_graph_compute(model, tailgf);
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_last_convnext" + std::to_string(j);
            PUSH_GGML_TRACE({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(tailgf, name.c_str()))});
        }
        PUSH_GGML_TRACE({"ve_proj_out", {L, Cin}, tensor_to_time_channel(ggml_graph_get_tensor(tailgf, "ve_proj_out"))});
        std::vector<float> next_latent_tc = tensor_to_time_channel(ggml_graph_get_tensor(tailgf, "ve_next_latent_tc"));
        if (next_latent_tc_out) *next_latent_tc_out = next_latent_tc;
        PUSH_GGML_TRACE({"ve_next_latent_tc", {L, Cin}, next_latent_tc});
        ggml_gallocr_free(tailallocr);
        ggml_gallocr_free(g3allocr);
        ggml_gallocr_free(g2allocr);
        ggml_gallocr_free(g1allocr);
        ggml_gallocr_free(sallocr);

        ggml_gallocr_free(allocr);
        if (error) error->clear();
#undef PUSH_GGML_TRACE
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

bool supertonic_vector_step_ggml(const supertonic_model & model,
                                 const float * noisy_latent,
                                 int latent_len,
                                 const float * text_emb,
                                 int text_len,
                                 const float * style_ttl,
                                 const float * latent_mask,
                                 int current_step,
                                 int total_steps,
                                 std::vector<float> & next_latent_out,
                                 std::string * error) {
    try {
        std::vector<supertonic_trace_tensor> scalar_trace;
        std::vector<supertonic_trace_tensor> ggml_trace;
        std::vector<float> next_tc;
        if (!supertonic_vector_trace_proj_ggml(model, noisy_latent, text_emb, text_len,
                                               style_ttl, latent_mask, latent_len,
                                               current_step, total_steps,
                                               scalar_trace, ggml_trace, error,
                                               false, false, &next_tc)) {
            return false;
        }
        const int L = latent_len;
        const int C = model.hparams.latent_channels;
        if (next_tc.size() != (size_t)L*C) throw std::runtime_error("bad ve_next_latent_tc size");
        next_latent_out.assign((size_t)C*L, 0.0f);
        for (int c = 0; c < C; ++c) {
            for (int t = 0; t < L; ++t) {
                next_latent_out[(size_t)c*L + t] = next_tc[(size_t)t*C + c];
            }
        }
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
