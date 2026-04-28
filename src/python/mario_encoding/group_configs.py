GROUP_N3_PRIMARY = {
  "output_label": "group_n3_primary_analysis",
  "subjects": [
    {
      "subject": 1,
      "train_sessions": [7, 8, 9, 10, 12],
      "test_sessions": [14],
      "experiment_id": "20260317_124704"
    },
    {
      "subject": 3,
      "train_sessions": [11, 12, 13, 14, 15, 16],
      "test_sessions": [17],
      "experiment_id": "20260317_124821"
    },
    {
      "subject": 5,
      "train_sessions": [14, 16, 17],
      "test_sessions": [18],
      "experiment_id": "20260317_124821"
    }
  ],
  "metrics_to_average": [
    "R2_full",
    "R2_unique_motor",
    "R2_unique_scene",
    "R2_unique_actions",
    "R2_unique_activity",
    "R2_shared",
    "product_measure_motor",
    "product_measure_scene",
    "product_measure_actions",
    "product_measure_activity"
  ],
  "feature_spaces": {
    "motor": [
      "B",
      "A",
      "RIGHT",
      "LEFT",
      "UP",
      "DOWN"
    ],
    "actions": [
      "is_airborne",
      "moving_right",
      "moving_left",
      "screen_scrolling"
    ],
    "scene": [
      "Enemy",
      "2-Horde",
      "3-Horde",
      "4-Horde",
      "Roof",
      "Gap",
      "Multiple gaps",
      "Variable gaps",
      "Gap enemy",
      "Pillar gap",
      "Valley",
      "Pipe valley",
      "Empty valley",
      "Enemy valley",
      "Roof valley",
      "Stair up",
      "Stair down",
      "Empty stair valley",
      "Enemy stair valley",
      "Gap stair valley",
      "2-Path",
      "3-Path",
      "Risk/Reward",
      "Reward",
      "Moving platform",
      "Flagpole",
      "Beginning",
      "Bonus zone",
      "Waterworld",
      "Checkpoint",
      "mario_big",
      "mario_fire",
      "mario_star",
      "powerup_visible",
      "enemy_drawn19",
      "enemy_drawn17",
      "enemy_drawn18",
      "enemy_drawn16",
      "enemy_drawn15"
    ],
    "activity": [
      "event_brick_smashed",
      "event_coin_collected",
      "event_powerup_collected",
      "event_hit_life_lost",
      "event_hit_powerup_lost",
      "event_kill_stomp",
      "event_kill_impact",
      "event_kill_kick"
    ]
  },
}
