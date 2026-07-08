#!/usr/bin/env python3
# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors
# SPDX-FileCopyrightText: 2026 FIDAA contributors
#
# Licensed under the EUPL, Version 1.2 only (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#   https://eupl.eu/1.2/en/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Export all conversation steps + feedback to CSV.

Run:  docker compose exec app python export.py
Output: /app/export.csv (appears on host via volume mount)
"""

import csv
import os

import playhouse.db_url as ph_url


def main():
    db = ph_url.connect(os.getenv("DATABASE_URL", "sqlite:///users.db"))

    query = """
    SELECT
        t.id::text       AS thread_id,
        u.identifier     AS user_email,
        s.type           AS step_type,
        s.name           AS step_name,
        s.input          AS input,
        s.output         AS output,
        f.value          AS feedback_rating,
        f.comment        AS feedback_comment,
        s."startTime"    AS started_at,
        s."createdAt"    AS created_at
    FROM "Step" s
    LEFT JOIN "Thread" t   ON s."threadId" = t.id
    LEFT JOIN "User" u     ON t."userId" = u.id
    LEFT JOIN "Feedback" f ON f."stepId" = s.id
    ORDER BY t."createdAt", s."startTime"
    """

    cursor = db.execute_sql(query)
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    with open("/app/export.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)

    n_threads = len({r[0] for r in rows if r[0]})
    n_feedback = sum(1 for r in rows if r[6] is not None)
    print(f"Exported {len(rows)} steps from {n_threads} conversations "
          f"({n_feedback} with feedback) to /app/export.csv")

    db.close()


if __name__ == "__main__":
    main()
