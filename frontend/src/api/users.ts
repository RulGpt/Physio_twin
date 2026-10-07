import { apiClient } from "./client";

import type {
  ProfileCreateRequest,
  ProfileCreateResponse,
  ProfileLoginRequest,
  ProfileResponse,
  UserHistoryResponse,
  UserResponse,
} from "../types/api";


// =============================================================================
// SIGN UP
// =============================================================================

export async function createProfile(
  request: ProfileCreateRequest,
): Promise<ProfileCreateResponse> {
  const response = await apiClient.post<ProfileCreateResponse>(
    "/users",
    request,
  );

  return response.data;
}


// =============================================================================
// LOGIN
// =============================================================================

export async function loginProfile(
  request: ProfileLoginRequest,
): Promise<ProfileResponse> {
  const response = await apiClient.post<ProfileResponse>(
    "/users/login",
    request,
  );

  return response.data;
}


// =============================================================================
// GET USER
// =============================================================================

export async function getUser(
  userId: string,
): Promise<UserResponse> {
  const response = await apiClient.get<UserResponse>(
    `/users/${userId}`,
  );

  return response.data;
}


// =============================================================================
// GET USER HISTORY
// =============================================================================

export async function getUserHistory(
  userId: string,
): Promise<UserHistoryResponse> {
  const response = await apiClient.get<UserHistoryResponse>(
    `/users/${userId}/history`,
  );

  return response.data;
}