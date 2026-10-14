# ReadyNow! architecture

How a message flows: the root's input callbacks log it, screen it (local patterns + Model Armor) and check it is on-mission; the root agent then calls specialists as tools and writes one answer. Every agent logs each step as a JSON line, which Agent Platform sends to Cloud Logging. Google Maps (Places API (New) for geocoding and safe places, Routes API for directions) is called with Google Cloud OAuth, or with an API key when one is configured.

```mermaid
flowchart LR
    U([User])

    subgraph AP["Agent Platform (Agent Engine) — ReadyNow! deployment"]
        direction TB
        subgraph G["1 · Input callbacks on the root (before_model)"]
            direction LR
            L1["log_user_prompt"] --> S["screen_input<br/>length · local patterns · Model Armor"] --> C["check_on_mission<br/>Gemini classifier"]
        end
        R["2 · readynow_root — Gemini 2.5 Flash<br/>describes capabilities, plans, calls specialists, writes the answer"]
        subgraph SP["3 · Specialists, called as tools (AgentTool) · every agent logs via callbacks"]
            direction LR
            W["weather_agent<br/>get_active_alerts · get_forecast<br/>+ US-only gate (before_tool)"]
            RT["routes_agent<br/>find_safe_places · get_route<br/>+ US-only gate (before_tool)"]
            N["news_agent<br/>google_search"]
            subgraph Q["safety_qa — SequentialAgent"]
                direction TB
                D["safety_drafter<br/>google_search"] --> V["safety_validator"] --> F["safety_refiner"]
            end
        end
        C -->|allowed| R
        R --> W & RT & N & Q
    end

    subgraph EXT["Google Cloud and external services"]
        direction TB
        NWS[("NWS API<br/>alerts · forecasts")]
        MAPS[("Google Maps Platform<br/>Places (New) · Routes<br/>OAuth sign-in or API key")]
        GS[("Google Search")]
        MA[("Model Armor<br/>AgentAI_Course")]
        SM[("Secret Manager<br/>Maps API key, if used")]
        CL[("Cloud Logging<br/>JSON interaction log")]
    end

    U -->|question| L1
    S -.->|blocked| U
    C -.->|off-mission refusal| U
    R -->|answer| U
    W --> NWS
    W & RT --> MAPS
    RT -.->|optional key| SM
    N & D --> GS
    S --> MA
    AP -.->|stdout| CL
```

![ReadyNow! architecture](architecture.png)
