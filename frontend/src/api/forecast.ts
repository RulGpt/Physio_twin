import { apiClient } from "./client";

import type {
  ForecastRequest,
  ForecastResponse,
} from "../types/api";


// =============================================================================
// GENERATE FORECAST
// =============================================================================

export async function getForecast(
  userId: string,
  request: ForecastRequest,
): Promise<ForecastResponse> {
  if (request.user_id !== userId) {
    throw new Error(
      "Forecast request user_id does not match the requested user.",
    );
  }

  const response = await apiClient.post<ForecastResponse>(
    `/users/${userId}/forecast`,
    request,
  );

  return response.data;
}