import {
  Component,
  ElementRef,
  ViewChild,
  AfterViewChecked,
  signal,
  computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatService } from './chat.service';

export interface Message {
  role: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements AfterViewChecked {
  @ViewChild('messagesEnd') private messagesEnd!: ElementRef;

  readonly sessionId = crypto.randomUUID();

  messages = signal<Message[]>([{
    role: 'bot',
    text: '👋 Hola, soy tu asistente en Machine Learning con Scikit-Learn y TensorFlow. Pregúntame sobre algoritmos, modelos, código o conceptos. ¿En qué puedo ayudarte?',
    timestamp: new Date(),
  }]);

  inputText = '';
  loading = signal(false);

  /** Chat window state: 'open' | 'minimized' | 'closed' */
  windowState = signal<'open' | 'minimized' | 'closed'>('open');

  isOpen = computed(() => this.windowState() === 'open');
  isMinimized = computed(() => this.windowState() === 'minimized');
  isClosed = computed(() => this.windowState() === 'closed');

  private shouldScroll = false;

  constructor(private chatService: ChatService) {}

  ngAfterViewChecked(): void {
    if (this.shouldScroll) {
      this.scrollToBottom();
      this.shouldScroll = false;
    }
  }

  private scrollToBottom(): void {
    try {
      this.messagesEnd?.nativeElement.scrollIntoView({ behavior: 'smooth' });
    } catch {}
  }

  openChat(): void {
    this.windowState.set('open');
  }

  minimizeChat(): void {
    this.windowState.set('minimized');
  }

  maximizeChat(): void {
    this.windowState.set('open');
  }

  closeChat(): void {
    this.windowState.set('closed');
  }

  clearMessages(): void {
    this.messages.set([{
      role: 'bot',
      text: '🔄 Conversación reiniciada. Pregúntame lo que quieras sobre Machine Learning.',
      timestamp: new Date(),
    }]);
    this.chatService.resetSession(this.sessionId).subscribe();
  }

  sendMessage(): void {
    const text = this.inputText.trim();
    if (!text || this.loading()) return;

    this.messages.update(msgs => [
      ...msgs,
      { role: 'user', text, timestamp: new Date() },
    ]);
    this.inputText = '';
    this.loading.set(true);
    this.shouldScroll = true;

    this.chatService.sendMessage(this.sessionId, text).subscribe({
      next: (res) => {
        this.messages.update(msgs => [
          ...msgs,
          { role: 'bot', text: res.answer, timestamp: new Date() },
        ]);
        this.loading.set(false);
        this.shouldScroll = true;
      },
      error: () => {
        this.messages.update(msgs => [
          ...msgs,
          {
            role: 'bot',
            text: '❌ Error al conectar con el agente. Asegúrate de que el servidor Python está en marcha.',
            timestamp: new Date(),
          },
        ]);
        this.loading.set(false);
        this.shouldScroll = true;
      },
    });
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
    }
  }

  trackByIndex(index: number): number {
    return index;
  }
}
