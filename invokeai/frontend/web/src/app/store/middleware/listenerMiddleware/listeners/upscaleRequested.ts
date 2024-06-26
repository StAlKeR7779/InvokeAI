import { createAction } from '@reduxjs/toolkit';
import { logger } from 'app/logging/logger';
import type { AppStartListening } from 'app/store/middleware/listenerMiddleware';
import { parseify } from 'common/util/serialize';
import { buildAdHocUpscaleGraph } from 'features/nodes/util/graph/buildAdHocUpscaleGraph';
import { createIsAllowedToUpscaleSelector } from 'features/parameters/hooks/useIsAllowedToUpscale';
import { toast } from 'features/toast/toast';
import { t } from 'i18next';
import { queueApi } from 'services/api/endpoints/queue';
import type { BatchConfig, ImageDTO } from 'services/api/types';

export const upscaleRequested = createAction<{ imageDTO: ImageDTO }>(`upscale/upscaleRequested`);

export const addUpscaleRequestedListener = (startAppListening: AppStartListening) => {
  startAppListening({
    actionCreator: upscaleRequested,
    effect: async (action, { dispatch, getState }) => {
      const log = logger('session');

      const { imageDTO } = action.payload;
      const { image_name } = imageDTO;
      const state = getState();

      const { isAllowedToUpscale, detailTKey } = createIsAllowedToUpscaleSelector(imageDTO)(state);

      // if we can't upscale, show a toast and return
      if (!isAllowedToUpscale) {
        log.error(
          { imageDTO },
          t(detailTKey ?? 'parameters.isAllowedToUpscale.tooLarge') // should never coalesce
        );
        toast({
          id: 'NOT_ALLOWED_TO_UPSCALE',
          title: t(detailTKey ?? 'parameters.isAllowedToUpscale.tooLarge'), // should never coalesce
          status: 'error',
        });
        return;
      }

      const enqueueBatchArg: BatchConfig = {
        prepend: true,
        batch: {
          graph: buildAdHocUpscaleGraph({
            image_name,
            state,
          }),
          runs: 1,
        },
      };

      try {
        const req = dispatch(
          queueApi.endpoints.enqueueBatch.initiate(enqueueBatchArg, {
            fixedCacheKey: 'enqueueBatch',
          })
        );

        const enqueueResult = await req.unwrap();
        req.reset();
        log.debug({ enqueueResult: parseify(enqueueResult) }, t('queue.graphQueued'));
      } catch (error) {
        log.error({ enqueueBatchArg: parseify(enqueueBatchArg) }, t('queue.graphFailedToQueue'));

        if (error instanceof Object && 'status' in error && error.status === 403) {
          return;
        } else {
          toast({
            id: 'GRAPH_QUEUE_FAILED',
            title: t('queue.graphFailedToQueue'),
            status: 'error',
          });
        }
      }
    },
  });
};
