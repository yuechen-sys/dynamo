// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake optimistic lock manager.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;

use crate::SequenceHash;
use crate::object::ObjectLockManager;

use super::client::MooncakeObjectBlockClient;

/// Optimistic lock manager for Mooncake Store.
///
/// Mooncake  is unconditional overwrite (no conditional write semantics).
/// Since KVBM block writes are content-deterministically idempotent
/// (same  always maps to the same content), concurrent writes
/// to the same key by multiple workers produce identical results.
pub struct MooncakeLockManager {
    client: Arc<MooncakeObjectBlockClient>,
}

impl MooncakeLockManager {
    pub fn new(client: Arc<MooncakeObjectBlockClient>) -> Self {
        Self { client }
    }
}

impl ObjectLockManager for MooncakeLockManager {
    fn has_meta(
        &self, hash: SequenceHash
    ) -> BoxFuture<'static, Result<bool>> {
        let store = self.client.store.clone();
        let key = format!("{}.meta", self.client.format_key(&hash));

        Box::pin(async move {
            match store.is_exist(&key) {
                Ok(exists) => Ok(exists),
                Err(e) => {
                    tracing::warn!(key = %key, error = %e, "has_meta check failed");
                    Ok(false)
                }
            }
        })
    }

    fn try_acquire_lock(
        &self, _hash: SequenceHash
    ) -> BoxFuture<'static, Result<bool>> {
        // Optimistic lock: always succeed.
        Box::pin(async { Ok(true) })
    }

    fn create_meta(
        &self, hash: SequenceHash
    ) -> BoxFuture<'static, Result<()>> {
        let store = self.client.store.clone();
        let key = format!("{}.meta", self.client.format_key(&hash));

        Box::pin(async move {
            store.put(&key, b"1", None)
                .map_err(|e| anyhow::anyhow!("failed to create meta: {}", e))?;
            Ok(())
        })
    }

    fn release_lock(
        &self, _hash: SequenceHash
    ) -> BoxFuture<'static, Result<()>> {
        // Nothing to release for optimistic locking.
        Box::pin(async { Ok(()) })
    }
}
