-- SPDX-License-Identifier: EUPL-1.2-only
-- Copyright (C) 2026 FIDAA contributors
-- Licensed under the EUPL, Version 1.2 only. See: https://eupl.eu/1.2/en/

-- Chainlit ChainlitDataLayer schema
-- Matches: https://github.com/Chainlit/chainlit-datalayer/prisma/schema.prisma
-- Chainlit does NOT auto-create these tables; they must exist before first run.
-- The PostgreSQL image runs this file automatically on first volume init.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE "StepType" AS ENUM (
    'assistant_message', 'embedding', 'llm', 'retrieval', 'rerank',
    'run', 'system_message', 'tool', 'undefined', 'user_message'
);

CREATE TABLE IF NOT EXISTS "User" (
    "id"         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt"  TIMESTAMP NOT NULL DEFAULT NOW(),
    "updatedAt"  TIMESTAMP NOT NULL DEFAULT NOW(),
    "metadata"   JSONB NOT NULL DEFAULT '{}',
    "identifier" TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS "Thread" (
    "id"        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP NOT NULL DEFAULT NOW(),
    "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW(),
    "deletedAt" TIMESTAMP,
    "name"      TEXT,
    "metadata"  JSONB NOT NULL DEFAULT '{}',
    "tags"      TEXT[] DEFAULT '{}',
    "userId"    UUID REFERENCES "User"("id") ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS "Step" (
    "id"          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt"   TIMESTAMP NOT NULL DEFAULT NOW(),
    "updatedAt"   TIMESTAMP NOT NULL DEFAULT NOW(),
    "parentId"    UUID REFERENCES "Step"("id") ON DELETE CASCADE,
    "threadId"    UUID REFERENCES "Thread"("id") ON DELETE CASCADE,
    "input"       TEXT,
    "metadata"    JSONB NOT NULL DEFAULT '{}',
    "name"        TEXT,
    "output"      TEXT,
    "type"        "StepType" NOT NULL,
    "showInput"   TEXT DEFAULT 'json',
    "isError"     BOOLEAN DEFAULT FALSE,
    "startTime"   TIMESTAMP,
    "endTime"     TIMESTAMP,
    "command"     TEXT,
    "defaultOpen" BOOLEAN,
    "modes"       JSONB
);

CREATE TABLE IF NOT EXISTS "Element" (
    "id"          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt"   TIMESTAMP NOT NULL DEFAULT NOW(),
    "updatedAt"   TIMESTAMP NOT NULL DEFAULT NOW(),
    "threadId"    UUID REFERENCES "Thread"("id") ON DELETE CASCADE,
    "stepId"      UUID NOT NULL REFERENCES "Step"("id") ON DELETE CASCADE,
    "metadata"    JSONB NOT NULL DEFAULT '{}',
    "mime"        TEXT,
    "name"        TEXT NOT NULL,
    "objectKey"   TEXT,
    "url"         TEXT,
    "chainlitKey" TEXT,
    "display"     TEXT,
    "size"        TEXT,
    "language"    TEXT,
    "page"        INT,
    "props"       JSONB
);

CREATE TABLE IF NOT EXISTS "Feedback" (
    "id"        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    "createdAt" TIMESTAMP NOT NULL DEFAULT NOW(),
    "updatedAt" TIMESTAMP NOT NULL DEFAULT NOW(),
    "stepId"    UUID REFERENCES "Step"("id"),
    "name"      TEXT NOT NULL,
    "value"     FLOAT NOT NULL,
    "comment"   TEXT
);

CREATE INDEX IF NOT EXISTS "Element_stepId_idx"   ON "Element"("stepId");
CREATE INDEX IF NOT EXISTS "Element_threadId_idx" ON "Element"("threadId");
CREATE INDEX IF NOT EXISTS "Step_createdAt_idx"   ON "Step"("createdAt");
CREATE INDEX IF NOT EXISTS "Step_endTime_idx"     ON "Step"("endTime");
CREATE INDEX IF NOT EXISTS "Step_parentId_idx"    ON "Step"("parentId");
CREATE INDEX IF NOT EXISTS "Step_startTime_idx"   ON "Step"("startTime");
CREATE INDEX IF NOT EXISTS "Step_threadId_idx"    ON "Step"("threadId");
CREATE INDEX IF NOT EXISTS "Step_type_idx"        ON "Step"("type");
CREATE INDEX IF NOT EXISTS "Step_name_idx"        ON "Step"("name");
CREATE INDEX IF NOT EXISTS "Thread_createdAt_idx" ON "Thread"("createdAt");
CREATE INDEX IF NOT EXISTS "Thread_name_idx"      ON "Thread"("name");
