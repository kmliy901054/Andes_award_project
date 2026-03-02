// model_infer.c  (for EMGLSTM: LSTM(2->16) + FC(16->8->2) + Sigmoid)
//
// Expects model_weights.h to define:
//   LSTM0_W_IH (4H * I), LSTM0_W_HH (4H * H), LSTM0_B_IH (4H), LSTM0_B_HH (4H)
//   FC0_W (8 * 16), FC0_B (8)
//   FC2_W (2 * 8),  FC2_B (2)
//
// Input layout:
//   seq points to T timesteps, each timestep has MODEL_INPUT_SIZE floats.
//   seq[t * MODEL_INPUT_SIZE + k]

#include "model_infer.h"
#include "model_weights.h"
#include <math.h>
#include <stdint.h>

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE 2
#endif

#ifndef MODEL_HIDDEN_SIZE
#define MODEL_HIDDEN_SIZE 16
#endif

#ifndef MODEL_OUTPUT_SIZE
#define MODEL_OUTPUT_SIZE 2
#endif

// FC dims for EMGLSTM head
#ifndef MODEL_FC0_OUT
#define MODEL_FC0_OUT 8
#endif

static inline float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}
static inline float reluf(float x) { return x > 0.0f ? x : 0.0f; }

// ---- row-major access helpers ----
// LSTM0_W_IH: (4H, I)
static inline float LSTM_WIH(int row, int col) {
  return LSTM0_W_IH[row * MODEL_INPUT_SIZE + col];
}
// LSTM0_W_HH: (4H, H)
static inline float LSTM_WHH(int row, int col) {
  return LSTM0_W_HH[row * MODEL_HIDDEN_SIZE + col];
}

// FC0_W: (8, 16)
static inline float FC0W(int row, int col) {
  return FC0_W[row * MODEL_HIDDEN_SIZE + col];
}
// FC2_W: (2, 8)
static inline float FC2W(int row, int col) {
  return FC2_W[row * MODEL_FC0_OUT + col];
}

// PyTorch LSTM gate order in weight matrices: (i, f, g, o) stacked along dim0 (size 4H).
static inline void lstm_step(
    const float *x,     // [MODEL_INPUT_SIZE]
    float *h, float *c  // [H]
) {
  const int H = MODEL_HIDDEN_SIZE;

  // snapshot previous hidden so recurrent matmul uses h_{t-1} for all units
  float h_prev[MODEL_HIDDEN_SIZE];
  for (int k = 0; k < H; ++k) h_prev[k] = h[k];

  float h_new[MODEL_HIDDEN_SIZE];
  float c_new[MODEL_HIDDEN_SIZE];

  for (int i = 0; i < H; ++i) {
    const int row_i = i;
    const int row_f = i + H;
    const int row_g = i + 2 * H;
    const int row_o = i + 3 * H;

    // Start with both biases (PyTorch applies both; many implementations sum them)
    float gi = LSTM0_B_IH[row_i] + LSTM0_B_HH[row_i];
    float gf = LSTM0_B_IH[row_f] + LSTM0_B_HH[row_f];
    float gg = LSTM0_B_IH[row_g] + LSTM0_B_HH[row_g];
    float go = LSTM0_B_IH[row_o] + LSTM0_B_HH[row_o];

    // input contribution
    for (int k = 0; k < MODEL_INPUT_SIZE; ++k) {
      const float xv = x[k];
      gi += LSTM_WIH(row_i, k) * xv;
      gf += LSTM_WIH(row_f, k) * xv;
      gg += LSTM_WIH(row_g, k) * xv;
      go += LSTM_WIH(row_o, k) * xv;
    }

    // recurrent contribution
    for (int k = 0; k < H; ++k) {
      const float hv = h_prev[k];
      gi += LSTM_WHH(row_i, k) * hv;
      gf += LSTM_WHH(row_f, k) * hv;
      gg += LSTM_WHH(row_g, k) * hv;
      go += LSTM_WHH(row_o, k) * hv;
    }

    const float it = sigmoidf_fast(gi);
    const float ft = sigmoidf_fast(gf);
    const float gt = tanhf(gg);
    const float ot = sigmoidf_fast(go);

    const float ci = ft * c[i] + it * gt;
    const float hi2 = ot * tanhf(ci);

    c_new[i] = ci;
    h_new[i] = hi2;
  }

  for (int i = 0; i < H; ++i) {
    c[i] = c_new[i];
    h[i] = h_new[i];
  }
}

// API suggestion:
// If you previously had model_predict() returning float, now output is 2 dims.
// Keep BOTH functions if you want backward compatibility.
// - model_predict2(): writes 2 outputs
// - model_predict(): returns output[0] (optional convenience)

void model_predict(const float *seq, int T, float out2[2]) {
  const int H = MODEL_HIDDEN_SIZE;

  // ---- LSTM forward over time, take last hidden ----
  float h[MODEL_HIDDEN_SIZE] = {0};
  float c[MODEL_HIDDEN_SIZE] = {0};

  for (int t = 0; t < T; ++t) {
    const float *x = &seq[t * MODEL_INPUT_SIZE];
    lstm_step(x, h, c);
  }
  // Now h[] == PyTorch h[-1] for last timestep

  // ---- FC: 16 -> 8 (ReLU) ----
  float h0[MODEL_FC0_OUT];
  for (int j = 0; j < MODEL_FC0_OUT; ++j) {
    float acc = FC0_B[j];
    for (int k = 0; k < H; ++k) acc += FC0W(j, k) * h[k];
    h0[j] = reluf(acc);
  }

  // ---- FC: 8 -> 2 (Sigmoid) ----
  for (int o = 0; o < 2; ++o) {
    float acc = FC2_B[o];
    for (int j = 0; j < MODEL_FC0_OUT; ++j) acc += FC2W(o, j) * h0[j];
    out2[o] = sigmoidf_fast(acc);
  }
}
