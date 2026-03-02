#include <Arduino.h>
#include <math.h>
#include "model_infer.h"

// ---------------- Pins ----------------
const int PIN_EMG_LEFT  = A0;
const int PIN_EMG_RIGHT = A1;

// ---------------- High-rate sampling ----------------
const float FS_HZ = 1000.0f;
const uint32_t Ts_us = (uint32_t)(1000000.0f / FS_HZ);
uint32_t next_us = 0;

// ---------------- “110 rows in 5s” cadence ----------------
static constexpr uint32_t EMIT_NUM_US = 5000000UL;
static constexpr uint32_t EMIT_DEN    = 110;
static uint32_t emit_k = 0;
static uint32_t emit_start_us = 0;
static uint32_t next_emit_us = 0;

// ---------------- Model window ----------------
static constexpr int EMG_WINDOW = 40;
static float emg_ring[EMG_WINDOW][2];
static int ring_w = 0;
static bool ring_full = false;
static float seq[EMG_WINDOW * 2];

// ---------------- Digital filtering ----------------
const float FC_HP = 20.0f;
const float FC_LP = 450.0f;
const float NOTCH_F0 = 50.0f;
const float NOTCH_Q  = 20.0f;

// ---------------- MVC ----------------
float mvc_left  = 500.0f;
float mvc_right = 500.0f;
const float MVC_RAISE_ALPHA = 0.10f;
const float MVC_DECAY       = 0.9995f;
const float MVC_MIN         = 50.0f;

static inline float onepole_a(float fc, float fs) { return expf(-2.0f * PI * (fc / fs)); }

struct OnePoleHP { float a=0, y1=0, x1=0; float step(float x){ float y=a*(y1 + x - x1); x1=x; y1=y; return y; } };
struct OnePoleLP { float a=0, y1=0;         float step(float x){ y1 = a*y1 + (1.0f - a)*x; return y1; } };

float mean_dc_L = 0.0f, mean_dc_R = 0.0f;
const float MEAN_ALPHA = 0.001f;
static inline float remove_dc(float x, float &m){ m = (1.0f - MEAN_ALPHA) * m + MEAN_ALPHA * x; return x - m; }

struct Biquad {
  float b0=1,b1=0,b2=0,a1=0,a2=0,z1=0,z2=0;
  void reset(){ z1=z2=0; }
  void setNotch(float fs, float f0, float Q){
    float w0 = 2.0f * PI * (f0 / fs);
    float cw = cosf(w0), sw = sinf(w0);
    float alpha = sw / (2.0f * Q);
    float _b0=1.0f, _b1=-2.0f*cw, _b2=1.0f;
    float a0 = 1.0f + alpha;
    float _a1=-2.0f*cw, _a2=1.0f-alpha;
    b0=_b0/a0; b1=_b1/a0; b2=_b2/a0; a1=_a1/a0; a2=_a2/a0;
    reset();
  }
  float step(float x){
    float y = b0*x + z1;
    z1 = b1*x - a1*y + z2;
    z2 = b2*x - a2*y;
    return y;
  }
};

OnePoleHP hpL,hpR;
OnePoleLP lpL,lpR;
Biquad notchL,notchR;

static inline void mvc_update(float rect, float &mvc){
  mvc *= MVC_DECAY;
  if (rect > mvc) mvc = (1.0f - MVC_RAISE_ALPHA) * mvc + MVC_RAISE_ALPHA * rect;
  if (mvc < MVC_MIN) mvc = MVC_MIN;
}
static inline void wait_for_next_sample(uint32_t &next_us_local) {
  while ((int32_t)(micros() - next_us_local) < 0) {}
  next_us_local += Ts_us;
}
static inline uint32_t compute_emit_time(uint32_t start_us, uint32_t k) {
  uint64_t t = (uint64_t)(k + 1) * (uint64_t)EMIT_NUM_US;
  uint32_t offset = (uint32_t)(t / EMIT_DEN);
  return start_us + offset;
}

static inline void ring_push(float L, float R) {
  emg_ring[ring_w][0] = L;
  emg_ring[ring_w][1] = R;
  ring_w++;
  if (ring_w >= EMG_WINDOW) { ring_w = 0; ring_full = true; }
}
static inline void build_seq_from_ring() {
  int start = ring_full ? ring_w : 0;
  for (int t = 0; t < EMG_WINDOW; ++t) {
    int idx = (start + t) % EMG_WINDOW;
    seq[t*2 + 0] = emg_ring[idx][0];
    seq[t*2 + 1] = emg_ring[idx][1];
  }
}

// rate limit
static constexpr float IDLE_MAG_TH = 0.03f;
static constexpr uint32_t IDLE_PRINT_MS = 500;
static uint32_t last_idle_print_ms = 0;
static constexpr uint32_t ACTIVE_PRINT_MS = 100;
static uint32_t last_active_print_ms = 0;

void setup() {
  Serial.begin(115200);
  delay(200);

  // filters
  hpL.a = onepole_a(FC_HP, FS_HZ);
  lpL.a = onepole_a(FC_LP, FS_HZ);
  notchL.setNotch(FS_HZ, NOTCH_F0, NOTCH_Q);
  hpR.a = onepole_a(FC_HP, FS_HZ);
  lpR.a = onepole_a(FC_LP, FS_HZ);
  notchR.setNotch(FS_HZ, NOTCH_F0, NOTCH_Q);

  next_us = micros();
  emit_start_us = micros();
  emit_k = 0;
  next_emit_us = compute_emit_time(emit_start_us, emit_k);

  Serial.println("EMG inference boot ok");
  Serial.println("Output CSV: tmillis,Lp,Rp,imbalance,magnitude,finalL,finalR");
}

void loop() {
  // 1kHz sampling
  wait_for_next_sample(next_us);

  int rawL = analogRead(PIN_EMG_LEFT);
  int rawR = analogRead(PIN_EMG_RIGHT);
  float xL = (float)rawL;
  float xR = (float)rawR;

  xL = remove_dc(xL, mean_dc_L);
  xL = hpL.step(xL);
  xL = lpL.step(xL);
  xL = notchL.step(xL);

  xR = remove_dc(xR, mean_dc_R);
  xR = hpR.step(xR);
  xR = lpR.step(xR);
  xR = notchR.step(xR);

  mvc_update(fabsf(xL), mvc_left);
  mvc_update(fabsf(xR), mvc_right);

  float finalL = fabsf(xL) / mvc_left;
  float finalR = fabsf(xR) / mvc_right;

  uint32_t now_us = micros();
  if ((int32_t)(now_us - next_emit_us) >= 0) {
    while ((int32_t)(now_us - next_emit_us) >= 0) {
      ring_push(finalL, finalR);
      emit_k++;
      next_emit_us = compute_emit_time(emit_start_us, emit_k);
      if (emit_k > 1000000UL) {
        emit_start_us = micros();
        emit_k = 0;
        next_emit_us = compute_emit_time(emit_start_us, emit_k);
      }
    }
    if (!ring_full) return;

    build_seq_from_ring();
    float out2[2] = {0,0};
    model_predict(seq, EMG_WINDOW, out2);

    float Lp = out2[0];
    float Rp = out2[1];
    float imbalance = Lp - Rp;
    float magnitude = 0.5f * (Lp + Rp);

    uint32_t now_ms = millis();
    if (magnitude < IDLE_MAG_TH) {
      if (now_ms - last_idle_print_ms < IDLE_PRINT_MS) return;
      last_idle_print_ms = now_ms;
    } else {
      if (now_ms - last_active_print_ms < ACTIVE_PRINT_MS) return;
      last_active_print_ms = now_ms;
    }

    // ---- Build line ----
    char line[180];
    int n = snprintf(line, sizeof(line),
      "%lu,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
      (unsigned long)now_ms, Lp, Rp, imbalance, magnitude, finalL, finalR);

    (void)n;
    Serial.print(line);
  }
}
