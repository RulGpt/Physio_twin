import { useEffect, useMemo, useState } from "react";
import axios from "axios";

import {
  createProfile,
  getUserHistory,
  loginProfile,
} from "./api/users";

import { updateState } from "./api/state";

import { getForecast } from "./api/forecast";

import type {
  DailyRecord,
  ForecastResponse,
  ForecastDayInput,
  HistoryItem,
  ProfileCreateResponse,
  ProfileResponse,
} from "./types/api";

import "./App.css";


// =============================================================================
// TYPES
// =============================================================================

type AuthMode = "login" | "signup";

type DashboardTab =
  | "report"
  | "daily"
  | "history"
  | "forecast";

interface StoredUser {
  user_id: string;
  name: string;
  username: string;
}


// =============================================================================
// HELPERS
// =============================================================================

function todayString(): string {
  return new Date().toISOString().split("T")[0];
}


function formatDate(dateString: string): string {
  const date = new Date(`${dateString}T00:00:00`);

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}


function getErrorMessage(
  error: unknown,
  fallback: string,
): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => item.msg)
        .filter(Boolean)
        .join(", ");
    }
  }

  if (error instanceof Error && error.message) {
    return error.message;
  }

  return fallback;
}


// =============================================================================
// APP
// =============================================================================

function App() {

  // ===========================================================================
  // AUTH
  // ===========================================================================

  const [authMode, setAuthMode] =
    useState<AuthMode>("login");

  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const [user, setUser] =
    useState<StoredUser | null>(null);

  const [authLoading, setAuthLoading] =
    useState(false);

  const [authMessage, setAuthMessage] =
    useState("");

  const [error, setError] =
    useState("");


  // ===========================================================================
  // DASHBOARD
  // ===========================================================================

  const [activeTab, setActiveTab] =
    useState<DashboardTab>("report");

  const [history, setHistory] =
    useState<HistoryItem[]>([]);

  const [loadingHistory, setLoadingHistory] =
    useState(false);


  // ===========================================================================
  // DAILY LOGGING
  // ===========================================================================

  const [recordDate, setRecordDate] =
    useState(todayString());

  const [sleepHours, setSleepHours] =
    useState("8");

  const [sleepQuality, setSleepQuality] =
    useState("7");

  const [exerciseDuration, setExerciseDuration] =
    useState("0");

  const [exerciseIntensity, setExerciseIntensity] =
    useState("5");

  const [exerciseType, setExerciseType] =
    useState<
      DailyRecord["exercise"]["exercise_type"]
    >("none");

  const [stress, setStress] =
    useState("5");

  const [fatigue, setFatigue] =
    useState("5");

  const [recovery, setRecovery] =
    useState("5");

  const [savingRecord, setSavingRecord] =
    useState(false);

  const [recordMessage, setRecordMessage] =
    useState("");


  // ===========================================================================
  // FORECAST
  // ===========================================================================

  const [forecastDays, setForecastDays] =
    useState("7");

  const [forecastSleep, setForecastSleep] =
    useState("8");

  const [forecastExercise, setForecastExercise] =
    useState("30");

  const [forecastIntensity, setForecastIntensity] =
    useState("5");

  const [forecastStress, setForecastStress] =
    useState("5");

  const [forecast, setForecast] =
    useState<ForecastResponse | null>(null);

  const [forecastLoading, setForecastLoading] =
    useState(false);

  const [forecastMessage, setForecastMessage] =
    useState("");


  // ===========================================================================
  // RESTORE SESSION
  // ===========================================================================

  useEffect(() => {

    const storedUser =
      localStorage.getItem("physio_twin_user");

    if (!storedUser) {
      return;
    }

    try {

      const parsedUser =
        JSON.parse(storedUser) as StoredUser;

      if (
        parsedUser.user_id &&
        parsedUser.name &&
        parsedUser.username
      ) {
        setUser(parsedUser);
      }

    } catch {

      localStorage.removeItem(
        "physio_twin_user",
      );

    }

  }, []);


  // ===========================================================================
  // LOAD HISTORY
  // ===========================================================================

  async function loadHistory(
    userId: string,
  ) {

    setLoadingHistory(true);
    setError("");

    try {

      const response =
        await getUserHistory(userId);

      setHistory(response.history);

    } catch (err) {

      setHistory([]);

      setError(
        getErrorMessage(
          err,
          "Unable to load your records.",
        ),
      );

    } finally {

      setLoadingHistory(false);

    }
  }


  // ===========================================================================
  // LOGIN
  // ===========================================================================

  async function handleLogin(
    event: React.FormEvent,
  ) {

    event.preventDefault();

    setAuthLoading(true);
    setError("");
    setAuthMessage("");

    try {

      const response: ProfileResponse =
        await loginProfile({
          username,
          password,
        });

      const loggedInUser: StoredUser = {
        user_id: response.user_id,
        name: response.name,
        username: response.username,
      };

      localStorage.setItem(
        "physio_twin_user",
        JSON.stringify(loggedInUser),
      );

      setUser(loggedInUser);
      setPassword("");
      setActiveTab("report");

      await loadHistory(
        loggedInUser.user_id,
      );

    } catch (err) {

      setError(
        getErrorMessage(
          err,
          "Invalid username or password.",
        ),
      );

    } finally {

      setAuthLoading(false);

    }
  }


  // ===========================================================================
  // SIGN UP
  // ===========================================================================

  async function handleSignup(
    event: React.FormEvent,
  ) {

    event.preventDefault();

    setAuthLoading(true);
    setError("");
    setAuthMessage("");

    try {

      const response: ProfileCreateResponse =
        await createProfile({
          name,
          username,
          password,
        });

      setAuthMode("login");
      setPassword("");

      setAuthMessage(
        `Account created for ${response.username}. Please log in.`,
      );

    } catch (err) {

      setError(
        getErrorMessage(
          err,
          "Unable to create the account.",
        ),
      );

    } finally {

      setAuthLoading(false);

    }
  }


  // ===========================================================================
  // LOGOUT
  // ===========================================================================

  function handleLogout() {

    localStorage.removeItem(
      "physio_twin_user",
    );

    setUser(null);
    setHistory([]);
    setForecast(null);

    setUsername("");
    setPassword("");
    setName("");

    setError("");
    setAuthMessage("");

    setActiveTab("report");
  }


  // ===========================================================================
  // SAVE DAILY RECORD
  // ===========================================================================

  async function handleSaveDailyRecord(
    event: React.FormEvent,
  ) {

    event.preventDefault();

    if (!user) {
      return;
    }

    setSavingRecord(true);
    setRecordMessage("");
    setError("");

    const numericExerciseDuration =
      Number(exerciseDuration);

    const numericExerciseIntensity =
      exerciseType === "none"
        ? null
        : Number(exerciseIntensity);

    const dailyRecord: DailyRecord = {

      date: recordDate,

      sleep: {
        duration_hours: Number(sleepHours),
        quality: Number(sleepQuality),
      },

      exercise: {
        duration_minutes:
          numericExerciseDuration,

        intensity:
          numericExerciseIntensity,

        exercise_type:
          exerciseType,
      },

      subjective: {
        stress: Number(stress),
        fatigue: Number(fatigue),
        recovery: Number(recovery),
      },
    };

    try {

      await updateState({
        user_id: user.user_id,
        record: dailyRecord,
        reference_sleep_hours: 8,
      });

      await loadHistory(
        user.user_id,
      );

      setForecast(null);

      setRecordMessage(
        "Daily record saved successfully.",
      );

      setActiveTab("report");

    } catch (err) {

      setError(
        getErrorMessage(
          err,
          "Unable to save the daily record.",
        ),
      );

    } finally {

      setSavingRecord(false);

    }
  }


  // ===========================================================================
  // GENERATE FORECAST
  // ===========================================================================

  async function handleGenerateForecast(
    event: React.FormEvent,
  ) {

    event.preventDefault();

    if (!user) {
      return;
    }

    if (history.length === 0) {

      setForecastMessage(
        "Add at least one daily record before generating a forecast.",
      );

      return;
    }

    setForecastLoading(true);
    setForecastMessage("");
    setError("");

    try {

      const numberOfDays =
        Math.min(
          30,
          Math.max(
            1,
            Number(forecastDays),
          ),
        );

      const futureDays: ForecastDayInput[] =
        Array.from(
          { length: numberOfDays },
          () => ({
            sleep_hours:
              Number(forecastSleep),

            exercise_duration_minutes:
              Number(forecastExercise),

            exercise_intensity:
              Number(forecastIntensity),

            stress:
              Number(forecastStress),
          }),
        );

      const response =
        await getForecast(
          user.user_id,
          {
            user_id: user.user_id,
            future_days: futureDays,
            reference_sleep_hours: 8,
          },
        );

      setForecast(response);

    } catch (err) {

      setError(
        getErrorMessage(
          err,
          "Unable to generate forecast.",
        ),
      );

    } finally {

      setForecastLoading(false);

    }
  }


  // ===========================================================================
  // LATEST RECORD
  // ===========================================================================

  const latestRecord =
    useMemo(() => {

      if (history.length === 0) {
        return null;
      }

      return history[
        history.length - 1
      ];

    }, [history]);


  // ===========================================================================
  // AUTH SCREEN
  // ===========================================================================

  if (!user) {

    return (

      <div className="auth-page">

        <div className="auth-card">

          <div className="auth-brand">

            <div className="brand-title">
              PHYSIO-TWIN
            </div>

            <div className="brand-subtitle">
              Personal Physiological State Digital Twin
            </div>

          </div>


          <div className="auth-tabs">

            <button
              type="button"
              className={
                authMode === "login"
                  ? "auth-tab active"
                  : "auth-tab"
              }
              onClick={() => {

                setAuthMode("login");
                setError("");
                setAuthMessage("");

              }}
            >
              Login
            </button>


            <button
              type="button"
              className={
                authMode === "signup"
                  ? "auth-tab active"
                  : "auth-tab"
              }
              onClick={() => {

                setAuthMode("signup");
                setError("");
                setAuthMessage("");

              }}
            >
              Sign Up
            </button>

          </div>


          {authMode === "login" ? (

            <form
              className="auth-form"
              onSubmit={handleLogin}
            >

              <h2>
                Welcome back
              </h2>

              <p className="form-description">
                Log in to access your physiological report.
              </p>


              <label>
                Username

                <input
                  type="text"
                  value={username}
                  onChange={(event) =>
                    setUsername(
                      event.target.value,
                    )
                  }
                  placeholder="Enter username"
                  required
                />

              </label>


              <label>
                Password

                <input
                  type="password"
                  value={password}
                  onChange={(event) =>
                    setPassword(
                      event.target.value,
                    )
                  }
                  placeholder="Enter password"
                  required
                />

              </label>


              {authMessage && (
                <div className="success-message">
                  {authMessage}
                </div>
              )}


              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}


              <button
                type="submit"
                className="primary-button"
                disabled={authLoading}
              >
                {authLoading
                  ? "Logging in..."
                  : "Login"}
              </button>

            </form>

          ) : (

            <form
              className="auth-form"
              onSubmit={handleSignup}
            >

              <h2>
                Create account
              </h2>

              <p className="form-description">
                Create your PHYSIO-TWIN profile.
              </p>


              <label>
                Name

                <input
                  type="text"
                  value={name}
                  onChange={(event) =>
                    setName(
                      event.target.value,
                    )
                  }
                  placeholder="Your name"
                  required
                />

              </label>


              <label>
                Username

                <input
                  type="text"
                  value={username}
                  onChange={(event) =>
                    setUsername(
                      event.target.value,
                    )
                  }
                  placeholder="Choose a username"
                  required
                />

              </label>


              <label>
                Password

                <input
                  type="password"
                  value={password}
                  onChange={(event) =>
                    setPassword(
                      event.target.value,
                    )
                  }
                  placeholder="Minimum 8 characters"
                  minLength={8}
                  required
                />

              </label>


              {error && (
                <div className="error-message">
                  {error}
                </div>
              )}


              <button
                type="submit"
                className="primary-button"
                disabled={authLoading}
              >
                {authLoading
                  ? "Creating account..."
                  : "Create Account"}
              </button>

            </form>

          )}

        </div>

      </div>
    );
  }


  // ===========================================================================
  // DASHBOARD
  // ===========================================================================

  return (

    <div className="app">

      <header className="app-header">

        <div>

          <div className="brand-title">
            PHYSIO-TWIN
          </div>

          <div className="brand-subtitle">
            Personal Physiological State Digital Twin
          </div>

        </div>


        <div className="header-user">

          <div className="user-name">
            {user.name}
          </div>

          <div className="user-username">
            @{user.username}
          </div>

          <button
            type="button"
            className="logout-button"
            onClick={handleLogout}
          >
            Logout
          </button>

        </div>

      </header>


      <main className="dashboard">

        {/* ===================================================================
            NAVIGATION
        ==================================================================== */}

        <nav className="dashboard-nav">

          <button
            type="button"
            className={
              activeTab === "report"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() =>
              setActiveTab("report")
            }
          >
            Report
          </button>


          <button
            type="button"
            className={
              activeTab === "daily"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() => {

              setRecordMessage("");
              setError("");

              setActiveTab("daily");

            }}
          >
            Daily Logging
          </button>


          <button
            type="button"
            className={
              activeTab === "history"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() => {

              setError("");

              if (user) {
                void loadHistory(
                  user.user_id,
                );
              }

              setActiveTab("history");

            }}
          >
            Previous Records
          </button>


          <button
            type="button"
            className={
              activeTab === "forecast"
                ? "nav-button active"
                : "nav-button"
            }
            onClick={() => {

              setError("");
              setForecastMessage("");

              setActiveTab("forecast");

            }}
          >
            Forecast
          </button>

        </nav>


        {/* ===================================================================
            GLOBAL MESSAGES
        ==================================================================== */}

        {error && (
          <div className="error-message dashboard-message">
            {error}
          </div>
        )}


        {recordMessage && (
          <div className="success-message dashboard-message">
            {recordMessage}
          </div>
        )}


        {/* ===================================================================
            REPORT
        ==================================================================== */}

        {activeTab === "report" && (

          <>

            <section className="panel welcome-panel">

              <div>

                <h1>
                  Hello, {user.name}
                </h1>

                <p>
                  Here is your current physiological
                  state based on your recorded observations.
                </p>

              </div>

            </section>


            <section className="panel">

              <div className="section-heading">

                <div>

                  <h2>
                    Current State
                  </h2>

                  <p>
                    Latest latent state estimated by
                    the PHYSIO-TWIN state-space model.
                  </p>

                </div>


                {latestRecord && (

                  <div className="last-observed">

                    Last observed

                    <strong>
                      {formatDate(
                        latestRecord.date,
                      )}
                    </strong>

                  </div>

                )}

              </div>


              {!latestRecord ? (

                <div className="empty-state">

                  <div className="empty-state-title">
                    No observations available
                  </div>

                  <div className="empty-state-text">
                    Start by adding your first daily record.
                  </div>

                  <button
                    type="button"
                    className="primary-button small-button"
                    onClick={() =>
                      setActiveTab("daily")
                    }
                  >
                    Add Daily Record
                  </button>

                </div>

              ) : (

                <div className="state-grid">

                  <StateCard
                    title="Fatigue"
                    value={
                      latestRecord.state.fatigue
                    }
                    uncertainty={
                      latestRecord.state
                        .fatigue_uncertainty
                    }
                  />


                  <StateCard
                    title="Recovery"
                    value={
                      latestRecord.state.recovery
                    }
                    uncertainty={
                      latestRecord.state
                        .recovery_uncertainty
                    }
                  />


                  <StateCard
                    title="Load"
                    value={
                      latestRecord.state.load
                    }
                    uncertainty={
                      latestRecord.state
                        .load_uncertainty
                    }
                  />

                </div>

              )}

            </section>


            <section className="panel">

              <div className="section-heading">

                <div>

                  <h2>
                    Data Status
                  </h2>

                  <p>
                    Your longitudinal observations
                    currently available to the model.
                  </p>

                </div>

              </div>


              <div className="data-status-grid">

                <DataStat
                  label="Observations"
                  value={String(
                    history.length,
                  )}
                />


                <DataStat
                  label="User"
                  value={`@${user.username}`}
                />


                <DataStat
                  label="Model"
                  value="Linear state-space"
                />

              </div>

            </section>

          </>

        )}


        {/* ===================================================================
            DAILY LOGGING
        ==================================================================== */}

        {activeTab === "daily" && (

          <section className="panel">

            <div className="section-heading">

              <div>

                <h2>
                  Daily Logging
                </h2>

                <p>
                  Enter today's physiological observations.
                </p>

              </div>

            </div>


            <form
              className="daily-form"
              onSubmit={
                handleSaveDailyRecord
              }
            >

              <div className="form-section">

                <h3>
                  Date
                </h3>

                <label>

                  Record date

                  <input
                    type="date"
                    value={recordDate}
                    max={todayString()}
                    onChange={(event) =>
                      setRecordDate(
                        event.target.value,
                      )
                    }
                    required
                  />

                </label>

              </div>


              <div className="form-section">

                <h3>
                  Sleep
                </h3>

                <div className="form-grid">

                  <label>

                    Sleep duration (hours)

                    <input
                      type="number"
                      min="0"
                      max="24"
                      step="0.1"
                      value={sleepHours}
                      onChange={(event) =>
                        setSleepHours(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>


                  <label>

                    Sleep quality (1–10)

                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={sleepQuality}
                      onChange={(event) =>
                        setSleepQuality(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>

                </div>

              </div>


              <div className="form-section">

                <h3>
                  Exercise
                </h3>

                <div className="form-grid">

                  <label>

                    Exercise duration (minutes)

                    <input
                      type="number"
                      min="0"
                      max="1440"
                      step="1"
                      value={
                        exerciseDuration
                      }
                      onChange={(event) =>
                        setExerciseDuration(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>


                  <label>

                    Exercise type

                    <select
                      value={exerciseType}
                      onChange={(event) =>
                        setExerciseType(
                          event.target
                            .value as DailyRecord[
                              "exercise"
                            ][
                              "exercise_type"
                            ],
                        )
                      }
                    >

                      <option value="none">
                        None
                      </option>

                      <option value="walking">
                        Walking
                      </option>

                      <option value="running">
                        Running
                      </option>

                      <option value="cycling">
                        Cycling
                      </option>

                      <option value="strength">
                        Strength
                      </option>

                      <option value="sports">
                        Sports
                      </option>

                      <option value="yoga">
                        Yoga
                      </option>

                      <option value="other">
                        Other
                      </option>

                    </select>

                  </label>


                  <label>

                    Exercise intensity (1–10)

                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={
                        exerciseIntensity
                      }
                      onChange={(event) =>
                        setExerciseIntensity(
                          event.target.value,
                        )
                      }
                      disabled={
                        exerciseType === "none"
                      }
                      required={
                        exerciseType !== "none"
                      }
                    />

                  </label>

                </div>

              </div>


              <div className="form-section">

                <h3>
                  Subjective State
                </h3>

                <div className="form-grid">

                  <label>

                    Stress (1–10)

                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={stress}
                      onChange={(event) =>
                        setStress(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>


                  <label>

                    Fatigue (1–10)

                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={fatigue}
                      onChange={(event) =>
                        setFatigue(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>


                  <label>

                    Recovery (1–10)

                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={recovery}
                      onChange={(event) =>
                        setRecovery(
                          event.target.value,
                        )
                      }
                      required
                    />

                  </label>

                </div>

              </div>


              <div className="form-actions">

                <button
                  type="submit"
                  className="primary-button"
                  disabled={savingRecord}
                >
                  {savingRecord
                    ? "Saving..."
                    : "Save Daily Record"}
                </button>

              </div>

            </form>

          </section>

        )}


        {/* ===================================================================
            HISTORY
        ==================================================================== */}

        {activeTab === "history" && (

          <section className="panel">

            <div className="section-heading">

              <div>

                <h2>
                  Previous Records
                </h2>

                <p>
                  Historical observations and
                  inferred physiological states.
                </p>

              </div>


              {loadingHistory && (
                <div className="loading-text">
                  Loading...
                </div>
              )}

            </div>


            {history.length === 0 ? (

              <div className="empty-state">

                <div className="empty-state-title">
                  No records yet
                </div>

                <div className="empty-state-text">
                  Your daily observations will appear here.
                </div>

              </div>

            ) : (

              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>

                      <th>
                        Date
                      </th>

                      <th>
                        Sleep
                      </th>

                      <th>
                        Exercise
                      </th>

                      <th>
                        Stress
                      </th>

                      <th>
                        Fatigue
                      </th>

                      <th>
                        Recovery
                      </th>

                      <th>
                        Model Fatigue
                      </th>

                      <th>
                        Model Recovery
                      </th>

                      <th>
                        Load
                      </th>

                    </tr>

                  </thead>


                  <tbody>

                    {history
                      .slice()
                      .reverse()
                      .map((item) => (

                        <tr key={item.date}>

                          <td>
                            {formatDate(
                              item.date,
                            )}
                          </td>

                          <td>
                            {item.inputs
                              .sleep_duration_hours
                              .toFixed(1)}
                            h
                          </td>

                          <td>
                            {
                              item.inputs
                                .exercise_duration_minutes
                            }
                            min
                          </td>

                          <td>
                            {item.inputs.stress.toFixed(1)}
                          </td>

                          <td>
                            {
                              item.observation
                                .fatigue_observed
                            }
                          </td>

                          <td>
                            {
                              item.observation
                                .recovery_observed
                            }
                          </td>

                          <td>
                            {item.state.fatigue.toFixed(3)}
                          </td>

                          <td>
                            {item.state.recovery.toFixed(3)}
                          </td>

                          <td>
                            {item.state.load.toFixed(3)}
                          </td>

                        </tr>

                      ))}

                  </tbody>

                </table>

              </div>

            )}

          </section>

        )}


        {/* ===================================================================
            FORECAST
        ==================================================================== */}

        {activeTab === "forecast" && (

          <>

            <section className="panel">

              <div className="section-heading">

                <div>

                  <h2>
                    Physiological Forecast
                  </h2>

                  <p>
                    Explore a future scenario using
                    your current estimated physiological state.
                  </p>

                </div>

              </div>


              {history.length === 0 ? (

                <div className="empty-state">

                  <div className="empty-state-title">
                    Forecast unavailable
                  </div>

                  <div className="empty-state-text">
                    Add at least one daily observation
                    before generating a forecast.
                  </div>

                  <button
                    type="button"
                    className="primary-button small-button"
                    onClick={() =>
                      setActiveTab("daily")
                    }
                  >
                    Add Daily Record
                  </button>

                </div>

              ) : (

                <form
                  className="daily-form"
                  onSubmit={
                    handleGenerateForecast
                  }
                >

                  <div className="form-section">

                    <h3>
                      Future Scenario
                    </h3>

                    <p className="form-description">
                      These conditions are applied to
                      each future day in the scenario.
                    </p>

                    <div className="form-grid">

                      <label>

                        Forecast horizon (days)

                        <input
                          type="number"
                          min="1"
                          max="30"
                          value={forecastDays}
                          onChange={(event) =>
                            setForecastDays(
                              event.target.value,
                            )
                          }
                          required
                        />

                      </label>


                      <label>

                        Sleep per day (hours)

                        <input
                          type="number"
                          min="0"
                          max="24"
                          step="0.1"
                          value={forecastSleep}
                          onChange={(event) =>
                            setForecastSleep(
                              event.target.value,
                            )
                          }
                          required
                        />

                      </label>


                      <label>

                        Exercise per day (minutes)

                        <input
                          type="number"
                          min="0"
                          max="1440"
                          value={
                            forecastExercise
                          }
                          onChange={(event) =>
                            setForecastExercise(
                              event.target.value,
                            )
                          }
                          required
                        />

                      </label>


                      <label>

                        Exercise intensity (1–10)

                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={
                            forecastIntensity
                          }
                          onChange={(event) =>
                            setForecastIntensity(
                              event.target.value,
                            )
                          }
                          required
                        />

                      </label>


                      <label>

                        Stress (1–10)

                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={forecastStress}
                          onChange={(event) =>
                            setForecastStress(
                              event.target.value,
                            )
                          }
                          required
                        />

                      </label>

                    </div>

                  </div>


                  {forecastMessage && (
                    <div className="error-message">
                      {forecastMessage}
                    </div>
                  )}


                  <div className="form-actions">

                    <button
                      type="submit"
                      className="primary-button"
                      disabled={
                        forecastLoading
                      }
                    >
                      {forecastLoading
                        ? "Generating..."
                        : "Generate Forecast"}
                    </button>

                  </div>

                </form>

              )}

            </section>


            {forecast && (

              <section className="panel">

                <div className="section-heading">

                  <div>

                    <h2>
                      Forecast Results
                    </h2>

                    <p>
                      Forecast starting after the
                      latest observed date.
                    </p>

                  </div>


                  <div className="last-observed">

                    Last observed

                    <strong>
                      {formatDate(
                        forecast.last_observed_date,
                      )}
                    </strong>

                  </div>

                </div>


                <div className="state-grid">

                  {forecast.forecast.length > 0 && (

                    <>

                      <StateCard
                        title="Day 1 Fatigue"
                        value={
                          forecast.forecast[0]
                            .fatigue
                        }
                        uncertainty={
                          forecast.forecast[0]
                            .fatigue_uncertainty
                        }
                      />


                      <StateCard
                        title="Day 1 Recovery"
                        value={
                          forecast.forecast[0]
                            .recovery
                        }
                        uncertainty={
                          forecast.forecast[0]
                            .recovery_uncertainty
                        }
                      />


                      <StateCard
                        title="Day 1 Load"
                        value={
                          forecast.forecast[0]
                            .load
                        }
                        uncertainty={
                          forecast.forecast[0]
                            .load_uncertainty
                        }
                      />

                    </>

                  )}

                </div>


                <div className="table-wrapper forecast-table-wrapper">

                  <table>

                    <thead>

                      <tr>

                        <th>
                          Horizon
                        </th>

                        <th>
                          Fatigue
                        </th>

                        <th>
                          Recovery
                        </th>

                        <th>
                          Load
                        </th>

                        <th>
                          Fatigue Uncertainty
                        </th>

                        <th>
                          Recovery Uncertainty
                        </th>

                        <th>
                          Load Uncertainty
                        </th>

                      </tr>

                    </thead>


                    <tbody>

                      {forecast.forecast.map(
                        (day) => (

                          <tr
                            key={day.horizon}
                          >

                            <td>
                              Day {day.horizon}
                            </td>

                            <td>
                              {day.fatigue.toFixed(3)}
                            </td>

                            <td>
                              {day.recovery.toFixed(3)}
                            </td>

                            <td>
                              {day.load.toFixed(3)}
                            </td>

                            <td>
                              ±{" "}
                              {day.fatigue_uncertainty.toFixed(
                                3,
                              )}
                            </td>

                            <td>
                              ±{" "}
                              {day.recovery_uncertainty.toFixed(
                                3,
                              )}
                            </td>

                            <td>
                              ±{" "}
                              {day.load_uncertainty.toFixed(
                                3,
                              )}
                            </td>

                          </tr>

                        ),
                      )}

                    </tbody>

                  </table>

                </div>


                <div className="forecast-validity">

                  <h3>
                    Forecast Validity
                  </h3>

                  {forecast.forecast.length > 0 && (

                    <div className="data-status-grid">

                      <DataStat
                        label="Fatigue"
                        value={
                          forecast.forecast[0]
                            .validity.fatigue.status
                        }
                      />

                      <DataStat
                        label="Recovery"
                        value={
                          forecast.forecast[0]
                            .validity.recovery.status
                        }
                      />

                      <DataStat
                        label="Load"
                        value={
                          forecast.forecast[0]
                            .validity.load.status
                        }
                      />

                    </div>

                  )}

                </div>

              </section>

            )}

          </>

        )}

      </main>

    </div>
  );
}


// =============================================================================
// STATE CARD
// =============================================================================

interface StateCardProps {
  title: string;
  value: number;
  uncertainty: number;
}


function StateCard({
  title,
  value,
  uncertainty,
}: StateCardProps) {

  return (

    <div className="state-card">

      <div className="state-card-title">
        {title}
      </div>

      <div className="state-card-value">
        {value.toFixed(3)}
      </div>

      <div className="state-card-uncertainty">
        ± {uncertainty.toFixed(3)}
      </div>

      <div className="state-card-description">
        Estimated latent state
      </div>

    </div>
  );
}


// =============================================================================
// DATA STAT
// =============================================================================

interface DataStatProps {
  label: string;
  value: string;
}


function DataStat({
  label,
  value,
}: DataStatProps) {

  return (

    <div className="data-stat">

      <span className="data-stat-label">
        {label}
      </span>

      <strong className="data-stat-value">
        {value}
      </strong>

    </div>
  );
}


export default App;