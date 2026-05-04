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
                                       const float * latent_mask,
                                       int latent_len,
                                       std::vector<supertonic_trace_tensor> & scalar_trace,
                                       std::vector<supertonic_trace_tensor> & ggml_trace,
                                       std::string * error) {
    try {
        scalar_trace.clear();
        ggml_trace.clear();
        const int L = latent_len;
        const int Cin = model.hparams.latent_channels;
        const int C = 512;

        std::vector<float> in((size_t) L * Cin);
        for (int t = 0; t < L; ++t) {
            for (int c = 0; c < Cin; ++c) {
                in[(size_t) t * Cin + c] = noisy_latent[(size_t) c * L + t];
            }
        }
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

        std::vector<float> te = time_embedding(model, 0, 5);
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
        std::vector<float> te_host = time_embedding(model, 0, 5);
        ggml_backend_tensor_set(t_emb, te_host.data(), 0, te_host.size() * sizeof(float));
        std::vector<float> text_lc_host((size_t) text_len * 256);
        for (int c = 0; c < 256; ++c) {
            for (int t = 0; t < text_len; ++t) {
                text_lc_host[(size_t)c * text_len + t] = text_emb[(size_t)c * text_len + t];
            }
        }
        ggml_backend_tensor_set(text_in, text_lc_host.data(), 0, text_lc_host.size() * sizeof(float));
        ggml_backend_graph_compute(model.backend, gf);

        ggml_trace.push_back({"ve_latent_tc", {L, Cin}, in});
        ggml_trace.push_back({"ve_masked", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_masked"))});
        for (int j = 0; j < 4; ++j) {
            const std::string name = "ve_block0_convnext" + std::to_string(j);
            ggml_trace.push_back({name, {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, name.c_str()))});
        }
        ggml_trace.push_back({"ve_time_add0", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_time_add0"))});
        ggml_trace.push_back({"ve_block2_convnext0", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_block2_convnext0"))});
        std::vector<float> q_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_q"));
        std::vector<float> k_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_k"));
        std::vector<float> v_out = tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_v"));
        ggml_trace.push_back({"ve_attn0_q", {L, 256}, q_out});
        ggml_trace.push_back({"ve_attn0_k", {text_len, 256}, k_out});
        ggml_trace.push_back({"ve_attn0_v", {text_len, 256}, v_out});
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
        ggml_backend_graph_compute(model.backend, gf);
        ggml_trace.push_back({"ve_attn0_q_rope", {L, 256}, q_out});
        ggml_trace.push_back({"ve_attn0_k_rope", {text_len, 256}, k_out});
        ggml_trace.push_back({"ve_attn0_ctx", {L, 256}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_ctx"))});
        ggml_trace.push_back({"ve_attn0_out", {L, C}, tensor_to_time_channel(ggml_graph_get_tensor(gf, "ve_attn0_out"))});

        ggml_gallocr_free(allocr);
        if (error) error->clear();
        return true;
    } catch (const std::exception & e) {
        if (error) *error = e.what();
        return false;
    }
}

} // namespace tts_cpp::supertonic::detail
