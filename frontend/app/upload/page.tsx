'use client'

import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, CheckCircle, XCircle, Loader2 } from 'lucide-react'
import { documentApi } from '@/lib/api'

export default function UploadPage() {
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<{
    status: 'idle' | 'uploading' | 'processing' | 'success' | 'error'
    message?: string
    documentId?: number
  }>({ status: 'idle' })

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return

    const file = acceptedFiles[0]
    
    setUploading(true)
    setUploadStatus({ status: 'uploading', message: 'Uploading file...' })

    try {
      const result = await documentApi.upload(file)
      
      setUploadStatus({
        status: 'processing',
        message: 'File uploaded. Processing document...',
        documentId: result.document_id
      })

      // Poll for status
      pollDocumentStatus(result.document_id)
      
    } catch (error: any) {
      setUploadStatus({
        status: 'error',
        message: error.response?.data?.detail || 'Upload failed'
      })
      setUploading(false)
    }
  }, [])

  const pollDocumentStatus = async (documentId: number) => {
    const maxAttempts = 60 // 5 minutes max
    let attempts = 0

    const poll = async () => {
      try {
        const status = await documentApi.getStatus(documentId)

        if (status.status === 'completed') {
          setUploadStatus({
            status: 'success',
            message: 'Document processed successfully!',
            documentId
          })
          setUploading(false)
        } else if (status.status === 'completed_with_warnings') {
          setUploadStatus({
            status: 'success',
            message: 'Document processed with warnings. Some tables may not have been extracted.',
            documentId
          })
          setUploading(false)
        } else if (status.status === 'failed') {
          setUploadStatus({
            status: 'error',
            message: status.error_message || 'Processing failed',
            documentId
          })
          setUploading(false)
        } else if (attempts < maxAttempts) {
          attempts++
          setTimeout(poll, 5000) // Poll every 5 seconds
        } else {
          setUploadStatus({
            status: 'error',
            message: 'Processing timeout',
            documentId
          })
          setUploading(false)
        }
      } catch (error) {
        setUploadStatus({
          status: 'error',
          message: 'Failed to check status',
          documentId
        })
        setUploading(false)
      }
    }

    poll()
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    disabled: uploading
  })

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Upload Fund Document</h1>
        <p className="text-gray-600">
          Upload a PDF fund performance report to automatically extract and analyze data
        </p>
      </div>

      {/* Upload Area */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition
          ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${uploading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="flex flex-col items-center">
          <Upload className="w-16 h-16 text-gray-400 mb-4" />
          
          {isDragActive ? (
            <p className="text-lg text-blue-600 font-medium">Drop the file here...</p>
          ) : (
            <>
              <p className="text-lg font-medium mb-2">
                Drag & drop a PDF file here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Maximum file size: 50MB
              </p>
            </>
          )}
        </div>
      </div>

      {/* Status Display */}
      {uploadStatus.status !== 'idle' && (
        <div className="mt-8">
          <div className={`
            rounded-lg p-6 border
            ${uploadStatus.status === 'success' ? 'bg-green-50 border-green-200' : ''}
            ${uploadStatus.status === 'error' ? 'bg-red-50 border-red-200' : ''}
            ${uploadStatus.status === 'uploading' || uploadStatus.status === 'processing' ? 'bg-blue-50 border-blue-200' : ''}
          `}>
            <div className="flex items-start">
              <div className="flex-shrink-0">
                {uploadStatus.status === 'success' && (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                )}
                {uploadStatus.status === 'error' && (
                  <XCircle className="w-6 h-6 text-red-600" />
                )}
                {(uploadStatus.status === 'uploading' || uploadStatus.status === 'processing') && (
                  <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
                )}
              </div>
              
              <div className="ml-4 flex-1">
                <h3 className={`
                  font-medium
                  ${uploadStatus.status === 'success' ? 'text-green-900' : ''}
                  ${uploadStatus.status === 'error' ? 'text-red-900' : ''}
                  ${uploadStatus.status === 'uploading' || uploadStatus.status === 'processing' ? 'text-blue-900' : ''}
                `}>
                  {uploadStatus.status === 'uploading' && 'Uploading...'}
                  {uploadStatus.status === 'processing' && 'Processing...'}
                  {uploadStatus.status === 'success' && 'Success!'}
                  {uploadStatus.status === 'error' && 'Error'}
                </h3>
                
                <p className={`
                  mt-1 text-sm
                  ${uploadStatus.status === 'success' ? 'text-green-700' : ''}
                  ${uploadStatus.status === 'error' ? 'text-red-700' : ''}
                  ${uploadStatus.status === 'uploading' || uploadStatus.status === 'processing' ? 'text-blue-700' : ''}
                `}>
                  {uploadStatus.message}
                </p>

                {uploadStatus.status === 'success' && (
                  <div className="mt-4 flex gap-3">
                    <a
                      href="/chat"
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition text-sm"
                    >
                      Start Asking Questions
                    </a>
                    <a
                      href="/funds"
                      className="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition text-sm"
                    >
                      View Fund Dashboard
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-12 bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4">What happens after upload?</h2>
        <div className="space-y-4">
          <Step
            number="1"
            title="Document Parsing"
            description="The system uses Docling to extract document structure, identifying tables and text sections."
          />
          <Step
            number="2"
            title="Data Extraction"
            description="Tables containing capital calls, distributions, and adjustments are parsed and stored in the database."
          />
          <Step
            number="3"
            title="Vector Embedding"
            description="Text content is chunked and embedded for semantic search, enabling natural language queries."
          />
          <Step
            number="4"
            title="Ready to Query"
            description="Once processing is complete, you can ask questions about the fund using the chat interface."
          />
        </div>
      </div>
    </div>
  )
}

function Step({ number, title, description }: {
  number: string
  title: string
  description: string
}) {
  return (
    <div className="flex items-start">
      <div className="flex-shrink-0 w-8 h-8 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center font-semibold text-sm">
        {number}
      </div>
      <div className="ml-4">
        <h3 className="font-medium text-gray-900">{title}</h3>
        <p className="text-sm text-gray-600 mt-1">{description}</p>
      </div>
    </div>
  )
}
