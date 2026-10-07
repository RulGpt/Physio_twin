// -----------------------------------------------------------------------------
// PHYSIO-TWIN API Types
// -----------------------------------------------------------------------------
// These interfaces mirror the current FastAPI backend contract.
// No model calculations are performed in the frontend.
// -----------------------------------------------------------------------------


// =============================================================================
// AUTHENTICATION / USER
// =============================================================================

export interface ProfileCreateRequest {
  name: string;
  username: string;
  password: string;
}

export interface ProfileCreateResponse {
  user_id: string;
  name: string;
  username: string;
  created: boolean;
}

export interface ProfileLoginRequest {
  username: string;
  password: string;
}

export interface ProfileResponse {
  user_id: string;
  name: string;
  username: string;
  created_at: string;
}

export interface UserResponse {
  user_id: string;
  created_at: string;
}


// =============================================================================
// DAILY OBSERVATION / STATE UPDATE
// =============================================================================

export interface SleepRecord {
  duration_hours: number;
  quality: number;

  bedtime?: string | null;
  wake_time?: string | null;
  consistency?: number | null;
}

export interface ExerciseRecord {
  duration_minutes: number;
  intensity?: number | null;
  sessions?: number | null;

  exercise_type?:
    | "none"
    | "walking"
    | "running"
    | "cycling"
    | "strength"
    | "sports"
    | "yoga"
    | "other";
}

export interface SubjectiveIndices {
  stress: number;
  fatigue: number;
  recovery: number;
}

export interface DailyRecord {
  date: string;

  sleep: SleepRecord;

  exercise: ExerciseRecord;

  subjective: SubjectiveIndices;
}

export interface ObservationRequest {
  user_id: string;
  record: DailyRecord;
  reference_sleep_hours?: number;
}

export interface StateEstimateResponse {
  user_id: string;
  date: string;

  fatigue: number;
  recovery: number;
  load: number;

  fatigue_uncertainty: number;
  recovery_uncertainty: number;
  load_uncertainty: number;

  sleep_deficit: number;

  exercise_load_raw: number;
  exercise_load_model: number;

  stress: number;
}


// =============================================================================
// HISTORY
// =============================================================================

export interface HistoryState {
  fatigue: number;
  recovery: number;
  load: number;

  fatigue_uncertainty: number;
  recovery_uncertainty: number;
  load_uncertainty: number;
}

export interface HistoryInputs {
  sleep_duration_hours: number;
  sleep_quality: number;

  sleep_deficit: number;

  exercise_duration_minutes: number;
  exercise_intensity: number | null;

  exercise_load_raw: number;
  exercise_load_model: number;

  stress: number;
}

export interface HistoryObservation {
  fatigue_observed: number;
  recovery_observed: number;
}

export interface HistoryItem {
  date: string;

  inputs: HistoryInputs;

  observation: HistoryObservation;

  state: HistoryState;
}

export interface UserHistoryResponse {
  user_id: string;
  history: HistoryItem[];
}


// =============================================================================
// FORECAST
// =============================================================================

export interface ForecastDayInput {
  sleep_hours: number;

  exercise_duration_minutes: number;

  exercise_intensity?: number | null;

  stress: number;
}

export interface ForecastRequest {
  user_id: string;

  future_days: ForecastDayInput[];

  reference_sleep_hours?: number;
}


// =============================================================================
// FORECAST VALIDITY
// =============================================================================

export type ForecastValidityStatus =
  | "within_scale"
  | "below_scale"
  | "above_scale"
  | "invalid_numeric";

export interface ForecastValidity {
  value: number;
  status: ForecastValidityStatus;
}

export interface ForecastValidityResponse {
  fatigue: ForecastValidity;
  recovery: ForecastValidity;
  load: ForecastValidity;
}


// =============================================================================
// FORECAST RESPONSE
// =============================================================================

export interface ForecastState {
  horizon: number;

  fatigue: number;
  recovery: number;
  load: number;

  fatigue_uncertainty: number;
  recovery_uncertainty: number;
  load_uncertainty: number;

  validity: ForecastValidityResponse;
}

export interface ForecastResponse {
  user_id: string;
  last_observed_date: string;

  forecast: ForecastState[];
}