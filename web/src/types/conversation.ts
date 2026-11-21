import type { RTMClient } from 'agora-rtm';

/** Session bootstrap from GET /api/get_config (channel + tokens + agent identity). */
export interface AgoraTokenData {
  token: string;
  uid: string;
  channel: string;
  agentId?: string;
  appId?: string; // `app_id` returned by backend
  agentUid?: string; // `NEXT_PUBLIC_AGENT_UID`
  familyId?: string;
  memberId?: string;
}

export interface AgoraRenewalTokens {
  rtcToken: string;
  rtmToken: string;
}

export interface ConversationMessage {
  role: "user" | "assistant";
  text: string;
}

export interface ConversationComponentProps {
  agoraData: AgoraTokenData;
  rtmClient: RTMClient;
  onTokenWillExpire: (uid: string) => Promise<AgoraRenewalTokens>;
  onEndConversation: (messages: ConversationMessage[]) => void;
}
