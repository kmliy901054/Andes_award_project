// model_infer.h
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// EMGLSTM outputs 2 values (left/right)
void model_predict(const float *seq, int T, float out2[2]);

#ifdef __cplusplus
}
#endif
