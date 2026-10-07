import { apiClient } from "./client";

import type {
  ObservationRequest,
  StateEstimateResponse,
} from "../types/api";


// =============================================================================
// UPDATE PHYSIOLOGICAL STATE
// =============================================================================
// This endpoint:
//   1. accepts a DailyRecord
//   2. loads the previous latent state
//   3. runs the Kalman update
//   4. persists the daily record
//   5. persists the inferred state
//   6. returns the updated state
// =============================================================================

export async function updateState(
  request: ObservationRequest,
): Promise<StateEstimateResponse> {
  const response = await apiClient.post<StateEstimateResponse>(
    "/state/update",
    request,
  );

  return response.data;
}