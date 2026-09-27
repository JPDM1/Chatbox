import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  answer: string;
  session_id: string;
}

@Injectable({ providedIn: 'root' })
export class ChatService {
  private readonly BASE = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  sendMessage(sessionId: string, message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.BASE}/chat`, {
      session_id: sessionId,
      message,
    });
  }

  resetSession(sessionId: string): Observable<unknown> {
    return this.http.post(`${this.BASE}/reset`, { session_id: sessionId });
  }
}
